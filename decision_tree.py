# import requests

import numpy as np
import pandas as pd
import ydf


def main():
    path = "https://storage.googleapis.com/download.tensorflow.org/data/palmer_penguins/penguins.csv"
    filename = path.split("/")[-1]
    # with requests.get(path, allow_redirects=True, stream=True) as r:
    #     r.raise_for_status()
    #
    #     with open(filename, "wb") as f:
    #         for chunk in r.iter_content(chunk_size=1024):
    #             f.write(chunk)
    dataset = pd.read_csv(filename)
    label = "species"

    print(dataset.head(3))
    print()

    # Use the ~20% of the examples as the testing set
    # and the remaining ~80% of the examples as the training set.
    np.random.seed(1)
    is_test = np.random.rand(len(dataset)) < 0.2

    train_dataset = dataset[~is_test]
    test_dataset = dataset[is_test]

    print("Training examples: ", len(train_dataset))
    # >> Training examples: 272

    print("Testing examples: ", len(test_dataset))
    # >> Testing examples: 72

    print()

    model = ydf.CartLearner(label=label).train(train_dataset)
    print()

    model.print_tree()
    print()
    print(model.describe())
    print()

    train_evaluation = model.evaluate(train_dataset)
    print("train accuracy:", train_evaluation.accuracy)
    # >> train accuracy:  0.9338

    test_evaluation = model.evaluate(test_dataset)
    print("test accuracy:", test_evaluation.accuracy)
    # >> test accuracy:  0.9167

    print()

    model = ydf.CartLearner(label=label, min_examples=1).train(train_dataset)
    print()

    model.print_tree()
    print()
    print(model.describe())
    print()

    train_evaluation = model.evaluate(train_dataset)
    print("train accuracy:", train_evaluation.accuracy)

    test_evaluation = model.evaluate(test_dataset)
    print("test accuracy:", test_evaluation.accuracy)

    print()

    model = ydf.RandomForestLearner(label=label).train(train_dataset)
    print()

    model.print_tree()
    print()
    print(model.describe())
    print()

    train_evaluation = model.evaluate(train_dataset)
    print("train accuracy:", train_evaluation.accuracy)

    test_evaluation = model.evaluate(test_dataset)
    print("test accuracy:", test_evaluation.accuracy)


if __name__ == "__main__":
    main()
