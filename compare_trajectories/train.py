from compare_trajectories.autoencoder import ConvAutoencoder
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split
import tensorflow as tf

logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
images_path = os.path.join(path,"data/images")

EPOCHS = 20
BATCH_SIZE = 32
DATASET_SIZE = 10**5


def train_autoencoder():
    data = []
    list_ = os.listdir(images_path)

    logging.info('Loading Data')

    for file in tqdm(list_):
        img = cv2.imread(os.path.join(images_path, file))
        img = cv2.resize(img, (64, 64))
        # img = np.expand_dims(img, 0)
        data.append(img)

    data = np.array(data)

    X_train, X_test, _, _ = train_test_split(
        data,
        data,
        test_size=0.25,
        random_state=42
    )

    # X_train = np.expand_dims(X_train, axis=-1)
    # X_test = np.expand_dims(X_test, axis=-1)
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    model = ConvAutoencoder(
        width=64,
        height=64,
        depth=3,
    )

    logging.info("Training Step")

    history = model.autoencoder.fit(
        X_train,
        X_train,
        validation_data=(X_test, X_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    model.autoencoder.save(os.path.join(path, 'data/autoencoder'))
    return history, model


def training_plot(history):
    N = np.arange(0, EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, history.history["loss"], label="train_loss")
    plt.plot(N, history.history["val_loss"], label="val_loss")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(path,"results","training_results.png"))


def prediction_plot(model,data,samples_number):
    decoded = model.predict(data)
    outputs = None
    for i in range(0, samples_number):
        image = (data[i] * 255).astype("uint8")
        predicted = (decoded[i] * 255).astype("uint8")
        output = np.hstack([image, predicted])
        if outputs is None:
            outputs = output
        else:
            outputs = np.vstack([outputs, output])
    cv2.imwrite(os.path.join(path,"results","predictions.png"), outputs)
