from __future__ import division
from __future__ import print_function

import glob
import os
import argparse
import cv2
import editdistance
from src.DataLoader import DataLoader, Batch
from src.Model import Model, DecoderType
from src.SamplePreprocessor import preprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print(BASE_DIR)


class FilePaths:
    """filenames and paths to data"""
    fnCharList = os.path.join(BASE_DIR, 'model/charList.txt')
    fnAccuracy = os.path.join(BASE_DIR, 'model/accuracy.txt')
    fnTrain = os.path.join(BASE_DIR, 'data/')
    fnInfer = os.path.join(BASE_DIR, 'data/test_images')
    fnCorpus = os.path.join(BASE_DIR, 'data/corpus.txt')


def train(model, loader):
    """train NN"""
    # number of training epochs since start
    epoch = 0
    best_char_error_rate = float('inf')  # best valdiation character error rate
    no_improvement_since = 0  # number of epochs no improvement of character error rate occured
    early_stopping = 5  # stop training after this number of epochs without improvement
    while True:
        epoch += 1
        print('Epoch:', epoch)

        # train
        print('Train NN')
        loader.train_set()
        while loader.has_next():
            iter_info = loader.get_iterator_info()
            batch = loader.get_next()
            loss = model.train_batch(batch)
            print('Batch:', iter_info[0], '/', iter_info[1], 'Loss:', loss)

        # validate
        char_error_rate = validate(model, loader)

        # if best validation accuracy so far, save model parameters
        if char_error_rate < best_char_error_rate:
            print('Character error rate improved, save model')
            best_char_error_rate = char_error_rate
            no_improvement_since = 0
            model.save()
            open(FilePaths.fnAccuracy, 'w').write(
                'Validation character error rate of saved model: %f%%' % (char_error_rate * 100.0))
        else:
            print('Character error rate not improved')
            no_improvement_since += 1

        # stop training if no more improvement in the last x epochs
        if no_improvement_since >= early_stopping:
            print('No more improvement since %d epochs. Training stopped.' % early_stopping)
            break


def validate(model, loader):
    """validate NN"""
    print('Validate NN')
    loader.validation_set()
    num_char_err = 0
    num_char_total = 0
    num_word_ok = 0
    num_word_total = 0
    while loader.has_next():
        iter_info = loader.get_iterator_info()
        print('Batch:', iter_info[0], '/', iter_info[1])
        batch = loader.get_next()
        (recognized, _) = model.infer_batch(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            num_word_ok += 1 if batch.gtTexts[i] == recognized[i] else 0
            num_word_total += 1
            dist = editdistance.eval(recognized[i], batch.gtTexts[i])
            num_char_err += dist
            num_char_total += len(batch.gtTexts[i])
            print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gtTexts[i] + '"', '->',
                  '"' + recognized[i] + '"')

    # print validation result
    char_error_rate = num_char_err / num_char_total
    word_accuracy = num_word_ok / num_word_total
    print('Character error rate: %f%%. Word accuracy: %f%%.' % (char_error_rate * 100.0, word_accuracy * 100.0))
    return char_error_rate


def infer(model, fn_img):
    """recognize text in image provided by file path"""
    img = preprocess(cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE), Model.imgSize)
    batch = Batch(None, [img])
    (recognized, probability) = model.infer_batch(batch, True)
    print('Recognized:', '"' + recognized[0] + '"')
    print('Probability:', probability[0])
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.flip(img, 1)
    while True:
        cv2.imshow('img', img)
        k = cv2.waitKey(33)
        if k == 32:  # Space key to stop
            break
        else:
            print(k) if k != -1 else 1


def main():
    """main function"""
    # optional command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='train the NN', action='store_true')
    parser.add_argument('--validate', help='validate the NN', action='store_true')
    parser.add_argument('--infer', help='use best path decoding', action='store_true')
    parser.add_argument('--beamsearch', help='use beam search', action='store_true')
    parser.add_argument('--wordbeamsearch', help='use word beam search', action='store_true')
    parser.add_argument('--dump', help='dump output of NN to CSV file(s)', action='store_true')

    args = parser.parse_args()

    decoder_type = DecoderType.BestPath
    if args.beamsearch:
        decoder_type = DecoderType.BeamSearch
    elif args.wordbeamsearch:
        decoder_type = DecoderType.WordBeamSearch

    # train or validate on IAM dataset
    if args.train or args.validate:
        # load training data, create TF model
        print(FilePaths.fnTrain)
        loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)

        # save characters of model for inference mode
        open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))

        # save words contained in dataset into file
        open(FilePaths.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))

        # execute training or validation
        if args.train:
            model = Model(loader.charList, decoder_type)
            train(model, loader)
        elif args.validate:
            model = Model(loader.charList, decoder_type, must_restore=True)
            validate(model, loader)

    # infer text on test image
    else:
        print(open(FilePaths.fnAccuracy).read())
        model = Model(open(FilePaths.fnCharList).read(), decoder_type, must_restore=True, dump=args.dump)
        image_files = [f for f in glob.glob(BASE_DIR + "/data/test_images/*.png")]
        print(image_files)
        for img in image_files:
            print(img)
            infer(model, img)


if __name__ == '__main__':
    main()
