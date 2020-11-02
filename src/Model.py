from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
import os


tf.compat.v1.disable_eager_execution()
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.get_logger().setLevel('ERROR')


class DecoderType:
    BestPath = 0
    BeamSearch = 1
    WordBeamSearch = 2


class Model:
    # minimalistic TF model for HTR
    # model constants
    batchSize = 50
    imgSize = (128, 32)
    maxTextLen = 32

    def __init__(self, char_list, decoder_type=DecoderType.BestPath, must_restore=False, dump=False):
        """init model: add CNN, RNN and CTC and initialize TF"""
        self.dump = dump
        self.charList = char_list
        self.decoderType = decoder_type
        self.mustRestore = must_restore
        self.snapID = 0

        # Whether to use normalization over a batch or a population
        self.is_train = tf.compat.v1.placeholder(tf.bool, name='is_train')

        # input image batch
        self.inputImgs = tf.compat.v1.placeholder(tf.float32, shape=(None, Model.imgSize[0], Model.imgSize[1]))

        # setup CNN, RNN and CTC
        self.setup_cnn()
        self.setup_rnn()
        self.setup_ctc()

        # setup optimizer to train NN
        self.batchesTrained = 0
        self.learningRate = tf.compat.v1.placeholder(tf.float32, shape=[])
        self.update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.optimizer = tf.compat.v1.train.RMSPropOptimizer(self.learningRate).minimize(self.loss)

        # initialize TF
        (self.sess, self.saver) = self.setup_tf()

    def setup_cnn(self):
        """create CNN layers and return output of these layers"""
        cnn_in4d = tf.expand_dims(input=self.inputImgs, axis=3)

        # list of parameters for the layers
        kernel_vals = [5, 5, 3, 3, 3]
        feature_vals = [1, 32, 64, 128, 128, 256]
        stride_vals = pool_vals = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2)]
        num_layers = len(stride_vals)

        # create layers
        pool = cnn_in4d  # input to first CNN layer
        for i in range(num_layers):
            kernel = tf.Variable(
                tf.random.truncated_normal(
                    [kernel_vals[i], kernel_vals[i], feature_vals[i], feature_vals[i + 1]],
                    stddev=0.1
                )
            )
            conv = tf.compat.v1.nn.conv2d(pool, kernel, padding='SAME', strides=(1, 1, 1, 1))
            conv_norm = tf.compat.v1.layers.batch_normalization(conv, training=self.is_train)
            relu = tf.compat.v1.nn.relu(conv_norm)
            pool = tf.compat.v1.nn.max_pool2d(
                relu,
                (1, pool_vals[i][0], pool_vals[i][1], 1),
                (1, stride_vals[i][0], stride_vals[i][1], 1),
                'VALID'
            )

        self.cnnOut4d = pool

    def setup_rnn(self):
        """create RNN layers and return output of these layers"""
        rnn_in3d = tf.squeeze(self.cnnOut4d, axis=[2])

        # basic cells which is used to build RNN
        num_hidden = 256
        cells = [tf.compat.v1.nn.rnn_cell.LSTMCell(
            num_units=num_hidden,
            state_is_tuple=True
        ) for _ in range(2)]  # 2 layers

        # stack basic cells
        stacked = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

        # bidirectional RNN
        # BxTxF -> BxTx2H
        ((fw, bw), _) = tf.compat.v1.nn.bidirectional_dynamic_rnn(
            cell_fw=stacked,
            cell_bw=stacked,
            inputs=rnn_in3d,
            dtype=rnn_in3d.dtype
        )

        # BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
        concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)

        # project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
        kernel = tf.Variable(
            tf.random.truncated_normal(
                [1, 1, num_hidden * 2,
                 len(self.charList) + 1],
                stddev=0.1
            )
        )
        self.rnnOut3d = tf.squeeze(
            tf.nn.atrous_conv2d(
                value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2]
        )

    def setup_ctc(self):
        """create CTC loss and decoder and return them"""
        # BxTxC -> TxBxC
        self.ctcIn3dTBC = tf.transpose(self.rnnOut3d, [1, 0, 2])
        # ground truth text as sparse tensor
        self.gtTexts = tf.SparseTensor(
            tf.compat.v1.placeholder(tf.int64, shape=[None, 2]),
            tf.compat.v1.placeholder(tf.int32, [None]),
            tf.compat.v1.placeholder(tf.int64, [2])
        )

        # calc loss for batch
        self.seqLen = tf.compat.v1.placeholder(tf.int32, [None])
        self.loss = tf.reduce_mean(
            tf.compat.v1.nn.ctc_loss(
                labels=self.gtTexts,
                inputs=self.ctcIn3dTBC,
                sequence_length=self.seqLen,
                ctc_merge_repeated=True
            )
        )

        # calc loss for each element to compute label probability
        self.savedCtcInput = tf.compat.v1.placeholder(
            tf.float32,
            shape=[Model.maxTextLen, None, len(self.charList) + 1]
        )
        self.lossPerElement = tf.compat.v1.nn.ctc_loss(
            labels=self.gtTexts,
            inputs=self.savedCtcInput,
            sequence_length=self.seqLen,
            ctc_merge_repeated=True
        )

        # decoder: either best path decoding or beam search decoding
        if self.decoderType == DecoderType.BestPath:
            self.decoder = tf.nn.ctc_greedy_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen)
        elif self.decoderType == DecoderType.BeamSearch:
            self.decoder = tf.nn.ctc_beam_search_decoder(
                inputs=self.ctcIn3dTBC,
                sequence_length=self.seqLen,
                beam_width=50,
                merge_repeated=False
            )
        elif self.decoderType == DecoderType.WordBeamSearch:
            # import compiled word beam search operation
            # (see https://github.com/githubharald/CTCWordBeamSearch)
            word_beam_search_module = tf.load_op_library('TFWordBeamSearch.so')

            # prepare information about language
            # (dictionary, characters in dataset, characters forming words)
            chars = str().join(self.charList)
            word_chars = open('../model/wordCharList.txt').read().splitlines()[0]
            corpus = open('../data/corpus.txt').read()

            # decode using the "Words" mode of word beam search
            self.decoder = word_beam_search_module.word_beam_search(
                tf.nn.softmax(self.ctcIn3dTBC, dim=2),
                50,
                'Words',
                0.0,
                corpus.encode('utf8'),
                chars.encode('utf8'),
                word_chars.encode('utf8')
            )

    def setup_tf(self):
        """initialize TF"""
        print('Python: ' + sys.version)
        print('Tensorflow: ' + tf.__version__)

        sess = tf.compat.v1.Session()  # TF session

        saver = tf.compat.v1.train.Saver(max_to_keep=1)  # saver saves model to file
        model_dir = 'model/'
        latest_snapshot = tf.train.latest_checkpoint(model_dir)  # is there a saved model?

        # if model must be restored (for inference), there must be a snapshot
        if self.mustRestore and not latest_snapshot:
            raise Exception('No saved model found in: ' + model_dir)

        # load saved model if available
        if latest_snapshot:
            print('Init with stored values from ' + latest_snapshot)
            saver.restore(sess, latest_snapshot)
        else:
            print('Init with new values')
            sess.run(tf.compat.v1.global_variables_initializer())

        return sess, saver

    def to_sparse(self, texts):
        indices = []
        values = []
        shape = [len(texts), 0]  # last entry must be max(labelList[i])

        # go over all texts
        for (batchElement, text) in enumerate(texts):
            # convert to string of label (i.e. class-ids)
            label_str = [self.charList.index(c) for c in text]
            # sparse tensor must have size of max. label-string
            if len(label_str) > shape[1]:
                shape[1] = len(label_str)
            # put each label into sparse tensor
            for (i, label) in enumerate(label_str):
                indices.append([batchElement, i])
                values.append(label)

        return indices, values, shape

    def decoder_output_to_text(self, ctc_output, batch_size):
        """extract texts from output of CTC decoder"""

        # contains string of labels for each batch element
        encoded_label_strs = [[] for i in range(batch_size)]

        # word beam search: label strings terminated by blank
        if self.decoderType == DecoderType.WordBeamSearch:
            blank = len(self.charList)
            for b in range(batch_size):
                for label in ctc_output[b]:
                    if label == blank:
                        break
                    encoded_label_strs[b].append(label)

        # TF decoders: label strings are contained in sparse tensor
        else:
            # ctc returns tuple, first element is SparseTensor
            decoded = ctc_output[0][0]

            # go over all indices and save mapping: batch -> values
            idx_dict = {b: [] for b in range(batch_size)}
            for (idx, idx2d) in enumerate(decoded.indices):
                label = decoded.values[idx]
                batch_element = idx2d[0]  # index according to [b,t]
                encoded_label_strs[batch_element].append(label)

        # map labels to chars for all batch elements
        return [str().join([self.charList[c] for c in labelStr]) for labelStr in encoded_label_strs]

    def train_batch(self, batch):
        """feed a batch into the NN to train it"""
        num_batch_elements = len(batch.imgs)
        sparse = self.to_sparse(batch.gtTexts)
        rate = 0.01 if self.batchesTrained < 10 else (
            0.001 if self.batchesTrained < 10000 else 0.0001)  # decay learning rate
        eval_list = [self.optimizer, self.loss]
        feed_dict = {
            self.inputImgs: batch.imgs,
            self.gtTexts: sparse,
            self.seqLen: [Model.maxTextLen] * num_batch_elements,
            self.learningRate: rate,
            self.is_train: True
        }
        (_, lossVal) = self.sess.run(eval_list, feed_dict)
        self.batchesTrained += 1
        return lossVal

    def dump_nn_output(self, rnn_output):
        """dump the output of the NN to CSV file(s)"""
        dump_dir = 'dump/'
        if not os.path.isdir(dump_dir):
            os.mkdir(dump_dir)

        # iterate over all batch elements and create a CSV file for each one
        maxT, maxB, maxC = rnn_output.shape
        for b in range(maxB):
            csv = ''
            for t in range(maxT):
                for c in range(maxC):
                    csv += str(rnn_output[t, b, c]) + ';'
                csv += '\n'
            fn = dump_dir + 'rnnOutput_' + str(b) + '.csv'
            print('Write dump of NN to file: ' + fn)
            with open(fn, 'w') as f:
                f.write(csv)

    def infer_batch(self, batch, calc_probability=False, probability_of_gt=False):
        """feed a batch into the NN to recognize the texts"""

        # decode, optionally save RNN output
        num_batch_elements = len(batch.imgs)
        eval_rnn_output = self.dump or calc_probability
        eval_list = [self.decoder] + ([self.ctcIn3dTBC] if eval_rnn_output else [])
        feed_dict = {
            self.inputImgs: batch.imgs,
            self.seqLen: [Model.maxTextLen] * num_batch_elements,
            self.is_train: False
        }
        eval_res = self.sess.run(eval_list, feed_dict)
        decoded = eval_res[0]
        texts = self.decoder_output_to_text(decoded, num_batch_elements)

        # feed RNN output and recognized text into CTC loss to compute labeling probability
        probs = None
        if calc_probability:
            sparse = self.to_sparse(batch.gtTexts) if probability_of_gt else self.to_sparse(texts)
            ctc_input = eval_res[1]
            eval_list = self.lossPerElement
            feed_dict = {
                self.savedCtcInput: ctc_input,
                self.gtTexts: sparse,
                self.seqLen: [Model.maxTextLen] * num_batch_elements,
                self.is_train: False
            }
            loss_vals = self.sess.run(eval_list, feed_dict)
            probs = np.exp(-loss_vals)

        # dump the output of the NN to CSV file(s)
        if self.dump:
            self.dump_nn_output(eval_res[1])

        return texts, probs

    def save(self):
        """save model to file"""
        self.snapID += 1
        self.saver.save(self.sess, 'model/snapshot', global_step=self.snapID)
