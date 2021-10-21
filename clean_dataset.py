import tensorflow as tf
import os
import glob


def remove_trash_files():
    files = []

    for file in glob.glob("dataset/**/.**", recursive=True):
        files.append(file)

    for file in glob.glob("dataset/**/*.jpg", recursive=True):
        try:
            img = tf.io.read_file(file)
            _img = tf.image.decode_jpeg(img)
        except:
            files.append(file)

    for file in files:
        os.remove(file)
        print(f"Remove file `{file}`")


if __name__ == "__main__":
    remove_trash_files()
