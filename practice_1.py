import os
import pickle

import keras
import matplotlib.pyplot as plt
import numpy as np


def main():
    with np.load("mnist.npz") as ds:
        x_train = ds["x_train"]
        y_train = ds["y_train"]
        x_test = ds["x_test"]
        y_test = ds["y_test"]

    x_train = x_train / 255.0
    y_train = keras.utils.to_categorical(y_train, 10)
    x_test = x_test / 255.0
    y_test = keras.utils.to_categorical(y_test, 10)

    # plt.figure(figsize=(10, 10))
    # for i in range(25):  # Display first 25 images
    #     plt.subplot(5, 5, i + 1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(
    #         x_train[i], cmap=plt.cm.binary
    #     )  # cmap=plt.cm.binary shows black on white
    #     plt.xlabel(f"Label: {y_train[i]}")
    # plt.suptitle("Sample MNIST Digits", fontsize=16)
    # plt.show()

    model_filepath = "model_1.keras"
    history_filepath = "history_1.pkl"
    if os.path.exists(model_filepath) and os.path.exists(history_filepath):
        model = keras.models.load_model(model_filepath)
        with open(history_filepath, "rb") as f:
            history = pickle.load(f)
    else:
        model = keras.models.Sequential(
            [
                keras.Input(shape=(28, 28)),
                keras.layers.Flatten(),
                keras.layers.Dense(
                    128,
                    activation="relu",
                    kernel_regularizer=keras.regularizers.l2(),
                ),
                keras.layers.Dense(
                    64,
                    activation="relu",
                    kernel_regularizer=keras.regularizers.l2(),
                ),
                keras.layers.Dense(
                    10,
                    activation="softmax",
                ),
            ]
        )

        model.compile(
            optimizer="adam",
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        model.summary()
        history = model.fit(
            x_train,
            y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
        )
        model.save(model_filepath)
        with open(history_filepath, "wb") as f:
            pickle.dump(history, f)

    model.evaluate(x_test, y_test, verbose=2)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(
        history.epoch,
        history.history["loss"],
        label="Loss",
    )
    plt.plot(
        history.epoch,
        history.history["val_loss"],
        label="Validation Loss",
    )
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(
        history.epoch,
        history.history["accuracy"],
        label="Accuracy",
    )
    plt.plot(
        history.epoch,
        history.history["val_accuracy"],
        label="Validation Accuracy",
    )
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    plt.plot()


if __name__ == "__main__":
    main()
