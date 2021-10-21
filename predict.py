import argparse
import glob
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from consts import IMAGE_SIZE, CLASSES
from nn import create_augmentation, create_model

LABELS = [
    "dark color, long tail",
    "dark color, short tail",
    "dog and owner",
    "empty",
    "light color, long tail",
    "light color, short tail",
    "multicolored, long tail",
    "multicolored, short tail",
    "other animal",
]
assert len(LABELS) == CLASSES


def show_predictions(predictions, image_path):
    score = predictions[0]
    percent = [category * 100 for category in score]

    plt.barh(LABELS, percent)
    plt.title(f"Predictions of {image_path}")
    plt.xlim(0, 100)
    plt.xlabel("Probability, percent")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.show()


def latest_checkpoint_path():
    checkpoints = glob.glob("checkpoints/*.h5")
    file = max(checkpoints, key=os.path.getctime)
    return file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", type=str, nargs="+")

    args = parser.parse_args()
    paths = args.paths

    augmentation = create_augmentation()
    model = create_model(augmentation)

    checkpoint = latest_checkpoint_path()
    model.load_weights(checkpoint)

    for img_path in paths:
        img = keras.preprocessing.image.load_img(
            img_path, target_size=IMAGE_SIZE
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(img_array)
        show_predictions(predictions, img_path)


if __name__ == "__main__":
    main()
