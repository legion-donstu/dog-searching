from tensorflow import keras
from tensorflow.keras import layers

from consts import IMAGE_SIZE, CLASSES
from nn.smaller_vgg_net import SmallerVGGNet
from nn.ctrl_c_callback import CtrlCStopping


def create_augmentation():
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            # layers.RandomRotation(0.01),
        ],
        name="augmentation",
    )


def create_model(augmentation):
    input_shape = IMAGE_SIZE + (3,)

    model = keras.Sequential()

    model.add(keras.Input(shape=input_shape))
    model.add(augmentation)
    model.add(layers.Rescaling(1.0 / 255))

    model.add(SmallerVGGNet(IMAGE_SIZE[0], IMAGE_SIZE[1], 3, CLASSES))

    return model
