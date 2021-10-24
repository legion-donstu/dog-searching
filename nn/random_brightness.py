import tensorflow as tf
from tensorflow.keras import layers


class RandomBrightness(layers.Layer):
    def __init__(self, max_delta):
        super(RandomBrightness, self).__init__()
        self.max_delta = max_delta

    def call(self, inputs, **kwargs):
        return tf.image.random_brightness(inputs, self.max_delta)
