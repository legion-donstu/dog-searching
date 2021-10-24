import tensorflow as tf
from tensorflow.keras import layers


class RandomHue(layers.Layer):
    def __init__(self, max_delta):
        super(RandomHue, self).__init__()
        self.max_delta = max_delta

    def call(self, inputs, **kwargs):
        return tf.image.random_hue(inputs, self.max_delta)
