from pathlib import Path

import tensorflow as tf
import logging

class WatermarkDetector(object):
    """wrapper for common functionality across watermark detection DNNs"""
    def __init__(self, model_trainer, input_size, *args, **kwargs):
        self._model = None
        self._model_trainer = model_trainer
        self._input_size = input_size
        return super().__init__(*args, **kwargs)

    def load(self, path):
         self._model = tf.keras.models.load_model(path)
         self._model.summary()

    def train(self, source_dataset, batch_size, path):
        dir = Path(path)
        dir.mkdir(0o777, True, True)

        # for now preprocessing and tf.data.Dataset is needed only for model based on effecient net
        # and for  other models fitting all training data to GPU greatly helps
        dataset = source_dataset.dataset().cache("dataset.cache")

        init_attempts = 1
        best_model = None
        best_loss = tf.constant(100.0, dtype=tf.float32)
        for init_attempt in range(init_attempts):
            logging.info("Training model: attempt %d", init_attempt)
            if dataset is None:
                model, losses = self._model_trainer(images, labels)
            else:
                model, losses = self._model_trainer(dataset)

            # save to log all metrics
            for epoch, loss in enumerate(losses):
                logging.info("Epoch %d, loss %.4f", epoch, losses[epoch])

            last_loss = losses[-1]
            if last_loss < best_loss:
                best_loss = last_loss
                best_model = model

        # recompile model with metrics needed for evaluation
        best_model.compile(metrics=[tf.keras.metrics.BinaryAccuracy()])

        best_model.save(path)
        logging.info("Best loss is %.6f", best_loss)

        self._model = best_model

        # in the end, check accuracy
        metrics = self._model.evaluate(dataset.batch(128), verbose = 0, return_dict=True)
        logging.info("Training accuracy is %0.1f%%", 100 * metrics["binary_accuracy"])

    def evaluate(self, source_dataset):
        dataset = None
        dataset = source_dataset.dataset().batch(128)
        metrics = self._model.evaluate(dataset, verbose = 0, return_dict=True)
        return metrics["binary_accuracy"]

    def __call__(self, image):
        resized = tf.image.resize(image, [self._input_size, self._input_size])

        # preprocessing data if procedure is not none
        data = tf.expand_dims(resized, 0)

        # now we feed the image to the model
        output = self._model(data)
        return output
