import matplotlib.pyplot as plt
import numpy as np


def main():
    with np.load("mnist.npz") as ds:
        raw_data = np.concatenate((ds["x_train"], ds["x_test"]))
        labels = np.concatenate((ds["y_train"], ds["y_test"]))

    data = raw_data.reshape((raw_data.shape[0], -1))
    print(data.shape)
    print(labels.shape)

    plt.hist(np.reshape(data, (-1,)), bins=17)
    plt.show()

    # flattened_mean = np.reshape(mean, (-1,))
    # plt.bar(np.arange(flattened_mean.size), flattened_mean)
    # plt.xticks(np.arange(0, flattened_mean.size, mean.shape[0]))
    # plt.show()
    #
    # flattened_transposed_mean = np.reshape(mean.transpose(), (-1,))
    # plt.bar(np.arange(flattened_transposed_mean.size), flattened_transposed_mean)
    # plt.xticks(np.arange(0, flattened_transposed_mean.size, mean.shape[0]))
    # plt.show()

    plt.imshow(np.mean(raw_data, axis=0))
    plt.show()

    plt.imshow(np.std(raw_data, axis=0))
    plt.show()


if __name__ == "__main__":
    main()
