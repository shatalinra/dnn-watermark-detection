import tensorflow as tf
import logging
from pathlib import Path

backbone = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
backbone.trainable = False

class StopIfFileExists(tf.keras.callbacks.Callback):
    """Callback that terminates training when certain file existss."""
    def __init__(self, filepath):
        self._filepath = Path(filepath)

    def on_batch_end(self, batch, logs=None):
        if self._filepath.is_file():
            self.model.stop_training = True


def preprocess(image):
    image = tf.image.resize(image, (224, 224))
    images = tf.expand_dims(image, 0)
    outputs =  backbone(255 * images, training = False)
    return outputs[0]

def train_model(dataset):
    model = tf.keras.Sequential(name = "efficient")
    model.add(tf.keras.layers.Dense(1, input_shape=(7,7,1280), name="dense")) # our data should be efficient embedding
    model.add(tf.keras.layers.Activation("sigmoid", name = "activation"))
    model.add(tf.keras.layers.GlobalMaxPool2D(name="pool")) # pick the location with most propability for watermark
    model.summary(print_fn=lambda x: logging.info(x))

    model.compile(loss=tf.losses.BinaryCrossentropy(), optimizer=tf.optimizers.Adam(learning_rate = 0.002))

    stop = StopIfFileExists('stop.txt')
    history = model.fit(dataset.batch(32), epochs=200, verbose=2, callbacks=[stop])
    return model, history.history["loss"]
