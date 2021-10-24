import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from consts import IMAGE_SIZE, CLASSES
from nn.random_brightness import RandomBrightness
from nn.random_hue import RandomHue
from nn.ctrl_c_callback import CtrlCStopping


def create_augmentation():
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            # layers.GaussianNoise(0.1),
            # layers.RandomContrast(0.2),
            RandomHue(0.1),
            RandomBrightness(0.1),
        ],
        name="augmentation",
    )


def create_model(augmentation):
    input_shape = IMAGE_SIZE + (3,)

    i = layers.Input(shape=input_shape, dtype=tf.uint8)
    # x = tf.cast(i, tf.float32)
    x = augmentation(i)
    x = keras.applications.efficientnet.preprocess_input(x)

    base_model = keras.applications.EfficientNetB1(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    base_model.trainable = False
    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(CLASSES, activation="softmax")(x)

    model = keras.Model(inputs=[i], outputs=[x])

    return model
