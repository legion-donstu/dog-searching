import glob
import os

import matplotlib.pyplot as plt
from tensorflow import keras

from consts import IMAGE_SIZE, BATCH_SIZE, BUFFER_SIZE, EPOCHS
from nn import CtrlCStopping, create_model, create_augmentation


def create_dataset(subset):
    ds = keras.preprocessing.image_dataset_from_directory(
        f"dataset",
        image_size=IMAGE_SIZE,
        label_mode="categorical",
        validation_split=0.2,
        seed=25432,
        subset=subset,
        batch_size=BATCH_SIZE,
    )
    ds = ds.prefetch(buffer_size=BUFFER_SIZE)
    return ds


def show_augmentation(dataset, augmentation):
    plt.suptitle("Data augmentation")

    for images, _ in dataset.take(1):
        for i in range(9):
            images = augmentation(images)
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(i)
            plt.axis("off")

    plt.tight_layout()
    plt.show()


def create_callbacks():
    os.makedirs("checkpoints", exist_ok=True)
    for h5 in glob.glob("checkpoints/*.h5"):
        os.remove(h5)

    return [
        CtrlCStopping(),
        keras.callbacks.ModelCheckpoint(
            "checkpoints/{epoch}.h5",
            save_weights_only=True,
        ),
        # keras.callbacks.EarlyStopping(
        #    monitor="val_loss",
        #    patience=20,
        # ),
    ]


def show_history(history):
    plt.figure(figsize=(12, 6))
    plt.suptitle("Learning history")

    plt.subplot(1, 2, 1)
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.plot(history.history["loss"], label="Training loss")
    plt.xlabel("Epochs")
    plt.xlim(right=EPOCHS)
    plt.ylabel("Loss")
    plt.ylim(bottom=0)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.plot(history.history["accuracy"], label="Training accuracy")
    plt.xlabel("Epochs")
    plt.xlim(right=EPOCHS)
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()

    plt.show()


def main():
    data_augmentation = create_augmentation()

    model = create_model(data_augmentation)
    keras.utils.plot_model(model, show_shapes=True)
    print(model.summary())

    train_ds = create_dataset("training")
    val_ds = create_dataset("validation")
    show_augmentation(train_ds, data_augmentation)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        callbacks=create_callbacks(),
        validation_data=val_ds,
    )
    show_history(history)


if __name__ == "__main__":
    main()
