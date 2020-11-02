import argparse
import glob
from action_methods import *


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
