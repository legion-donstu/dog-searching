import io

from tensorflow import keras
import tensorflow as tf
from typing import List

from PIL import Image
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, UploadFile, Form

from consts import IMAGE_SIZE
from nn import create_model, create_augmentation

LABELS = [
    ("dark", "long"),
    ("dark", "short"),
    "dog_and_owner",
    "empty",
    ("light", "long"),
    ("light", "short"),
    ("multicolored", "long"),
    ("multicolored", "short"),
    "other_animal",
]

augmentation = create_augmentation()
model = create_model(augmentation)
model.load_weights("checkpoints/1.h5")

app = FastAPI()


@app.post("/predictions")
def get_predictions(files: List[UploadFile] = Form(...)):
    images = []
    for upload_file in files:
        img = upload_file.file.read()
        img = Image.open(io.BytesIO(img))
        img = img.resize(IMAGE_SIZE)
        img = img.convert("RGB")
        img = keras.preprocessing.image.img_to_array(img)
        img = tf.expand_dims(img, 0)
        images.append(img)

    result = []

    for image in images:
        predictions = model.predict(image)
        score = predictions[0]

        predictions = []

        for i, percent in enumerate(score):
            label = LABELS[i]
            percent = int(percent * 100)
            if isinstance(label, tuple):
                color, tail = label
                predictions.append(
                    {
                        "type": "only_dog",
                        "color": color,
                        "tail_len": tail,
                        "percent": percent,
                    }
                )
            else:
                predictions.append({"type": label, "percent": percent})

        result.append(predictions)

    return result


@app.post("/prediction")
def predict(files: List[UploadFile] = Form(...)):
    result = get_predictions(files)
    result = [
        sorted(predictions, key=lambda obj: obj["percent"])[-1:][0]
        for predictions in result
    ]
    return result


app.mount("/", StaticFiles(directory="site", html=True))
