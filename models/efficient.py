import tensorflow as tf
import logging


backbone = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
backbone.trainable = False

def preprocess(image):
    image = tf.image.resize(image, (224, 224))
    images = tf.expand_dims(image, 0)
    outputs =  backbone(255 * images, training = False)
    return outputs[0]

def train_model(dataset):
    model = tf.keras.Sequential(name = "efficient")
    model.add(tf.keras.layers.GlobalAveragePooling2D(input_shape=(7,7,1280), name="pool")) # our data should be efficient embedding
    model.add(tf.keras.layers.Dense(1, name="dense"))
    model.add(tf.keras.layers.Activation("sigmoid", name = "activation"))
    model.summary(print_fn=lambda x: logging.info(x))

    model.compile(loss=tf.losses.BinaryCrossentropy(), optimizer=tf.optimizers.Adam(learning_rate = 0.002))

    history = model.fit(dataset, epochs=10, verbose=2)
    return model, history.history["loss"]
