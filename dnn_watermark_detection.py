import os, sys, logging, argparse, random

import tensorflow as tf
import matplotlib.pyplot as plt
import data
import watermark_detector
from models import efficient

# parse command line args
parser = argparse.ArgumentParser(description='Noise estimation DNN training script')
parser.add_argument('--log', help='Path to a log file.')
parser.add_argument('--validate', action='store_true', help='Validate model based on image separate from training or testing data.')
script_args = parser.parse_args()

# setup logging before anything else
log_format = '%(asctime)s: <%(levelname)s> %(message)s'
if script_args.log:
    try:
        error_stream = logging.StreamHandler()
        error_stream.setLevel(logging.INFO)
        log_file = logging.FileHandler(script_args.log)
        logging.basicConfig(format=log_format, level=logging.INFO, handlers=[error_stream, log_file])
    except OSError as err:
        print("Error while creating log {}: {}. Exiting...".format(err.filename, err.strerror))
        input("Press Enter to continue...")
        sys.exit(1)
else:
    logging.basicConfig(format=log_format, level=logging.INFO)

# now we can setup hooks for uncaught exceptions
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

# giving message that log is indeed initialized
print("Log initialized")

# common procedure for training, evaluating and validating model
def try_model(name, training_images, testing_images, step, batch_size, preprocessing, input_shape, model_trainer, validate):
    logging.info("Trying " + name + " model")
    detector = watermark_detector.WatermarkDetector(model_trainer, 224)
    try:
        # if everything will load fine we can go to testing the model
        detector.load("trained_models/" + name)

        if validate:
            # generate validation data
            clean_image = data.load_image('../coco/2017/train/000000001955.jpg')
            watermark = data.load_image("watermarks/0006.png")
            watermarked_image = data.apply_watermark(clean_image, 0, 0, watermark)

            # show original and noised image in order to check that noise generation is fine
            fig=plt.figure(figsize=(8, 2))
            fig.add_subplot(1, 3, 1)
            plt.imshow(clean_image)
            fig.add_subplot(1, 3, 2)
            plt.imshow(watermark)
            fig.add_subplot(1, 3, 3)
            plt.imshow(watermarked_image)
            plt.show()

            # run both cases though detector
            clean_data = clean_image
            watermarked_data = watermarked_image
            if preprocessing is not None:
                clean_data = preprocessing(clean_image)
                watermarked_data = preprocessing(watermarked_image)
            clean_confidence = detector(clean_data)
            watermarked_confidence = detector(watermarked_data)
            logging.info("Confidence in watermark precense on clean image is %.3f", clean_confidence)
            logging.info("Confidence in watermark precense on watermarked image is %.3f", watermarked_confidence)


        else:
            # testing the model
            # use CPU to support maybe longer but more representative evaluation on larger data
            # generate testing data from a portion of MS COCO 2017 train images
            dataset = data.WatermarkedImageDataset(testing_images[0], testing_images[1], step, preprocessing, input_shape)

            # now evaluate accuracy
            accuracy = detector.evaluate(dataset)
            logging.info("Testing accuracy is %0.1f%%", 100 * accuracy)

    except IOError:
        # looks like we don't have trained model, so we have to train one from scratch
        # but first generate data on CPU in order to leave GPU RAM for model, training variables and batches
        dataset = data.WatermarkedImageDataset(training_images[0], training_images[1], step, preprocessing, input_shape)

        #for image, labels in dataset.data_generator():
           # show original and noised image in order to check that noise generation is fine
         #  fig=plt.figure(figsize=(8, 2))
         #  fig.add_subplot(1, 2, 1)
         #  plt.imshow(image[0])
         #  plt.show()

        detector.train(dataset, batch_size, "trained_models/" + name)


training_images = (data.ImageSequence("../coco/2017/train/", 1, 50000), data.ImageSequence("watermarks/", 1, 4, digits = 4, extension="png"))
testing_images = (data.ImageSequence("../coco/2017/train/", 50000, 51000), data.ImageSequence("watermarks/", 1, 4, digits = 4, extension="png"))

try_model("efficent", training_images, testing_images, 10000, 64, efficient.preprocess, (7, 7, 1280), efficient.train_model, script_args.validate)