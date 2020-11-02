from __future__ import division
from __future__ import print_function

import os
import cv2
import editdistance
from src.DataLoader import DataLoader, Batch
from src.Model import Model, DecoderType
from src.SamplePreprocessor import preprocess
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print('Base DIR:', BASE_DIR)
print('Warning Disabled')


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


def cv_show(title, input_img):
    while True:
        cv2.imshow(title, input_img)
        k = cv2.waitKey(33)
        if k == 32:  # Space key to stop
            break
        else:
            print(k) if k != -1 else 1
    cv2.destroyWindow(title)


def plt_show(title, src_img, input_img):
    plt.suptitle(title)
    plt.subplot(221)
    plt.imshow(cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB))
    plt.title('Actual')
    inv_img = 255 - input_img
    plt.subplot(222)
    plt.imshow(inv_img, cmap='gray')
    plt.title('Model Input')
    plt.show()


def infer(model, fn_img, show_img=1):
    print("Image path: " + fn_img)
    """recognize text in image provided by file path"""
    src_img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
    input_img = preprocess(src_img, Model.imgSize)
    batch = Batch(None, [input_img])
    (recognized, probability) = model.infer_batch(batch, True)
    # print('Recognized:', '"' + recognized[0] + '"')
    # print('Probability:', probability[0])
    input_img = cv2.rotate(input_img, cv2.ROTATE_90_CLOCKWISE)
    input_img = cv2.flip(input_img, 1)

    title = 'Prediction: ' + recognized[0] + ', Probability: ' + (format(probability[0], ".2f"))
    if show_img == 1:
        cv_show(title, input_img)
    else:
        plt_show(title, src_img, input_img)


def infer_samples(decoder_type=DecoderType.BestPath):
    # print(open(FilePaths.fnAccuracy).read())
    model = Model(open(FilePaths.fnCharList).read(), decoder_type, must_restore=True)
    path = os.path.join('data', 'test_images')
    image_files = [os.path.join(path, f) for f in os.listdir(path)]
    # print(image_files)
    print("######### Inferring Text from Test Images###############")

    for img in image_files:
        # print(img)
        try:
            infer(model, img, show_img=2)
        except Exception as e:
            print("Exception: " + str(e))
            pass
