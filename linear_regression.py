import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from linear_regression.predict import predict_fare, show_predictions
from linear_regression.train import run_experiment

pd.options.mode.copy_on_write = True


def main():
    chicago_taxi_dataset = pd.read_csv("chicago_taxi_train.csv")
    training_df = chicago_taxi_dataset[
        ["TRIP_MILES", "TRIP_SECONDS", "FARE", "COMPANY", "PAYMENT_TYPE", "TIP_RATE"]
    ]

    # print("Read dataset completed successfully.")
    # print("Total number of rows: {0}".format(len(training_df.index)))
    # print()
    print(training_df.head())
    print()
    print(training_df.describe(include="all"))
    print()

    # What is the maximum fare?
    max_fare = training_df["FARE"].max()
    print("What is the maximum fare? \t\t\t\tAnswer: ${fare:.2f}".format(fare=max_fare))

    # What is the mean distance across all trips?
    mean_distance = training_df["TRIP_MILES"].mean()
    print(
        "What is the mean distance across all trips? \t\tAnswer: {mean:.4f} miles".format(
            mean=mean_distance
        )
    )

    # How many cab companies are in the dataset?
    num_unique_companies = training_df["COMPANY"].nunique()
    print(
        "How many cab companies are in the dataset? \t\tAnswer: {number}".format(
            number=num_unique_companies
        )
    )

    # What is the most frequent payment type?
    most_freq_payment_type = training_df["PAYMENT_TYPE"].value_counts().idxmax()
    print(
        "What is the most frequent payment type? \t\tAnswer: {type}".format(
            type=most_freq_payment_type
        )
    )

    # Are any features missing data?
    missing_values = training_df.isnull().sum().sum()
    print(
        "Are any features missing data? \t\t\t\tAnswer:",
        "No" if missing_values == 0 else "Yes",
    )

    print()

    print(training_df.corr(numeric_only=True))
    print()

    sns.pairplot(
        training_df,
        x_vars=["FARE", "TRIP_MILES", "TRIP_SECONDS"],
        y_vars=["FARE", "TRIP_MILES", "TRIP_SECONDS"],
    )
    plt.show()

    # The following variables are the hyperparameters.
    learning_rate = 0.001
    # learning_rate = 1.0
    # learning_rate = 0.0001
    epochs = 20
    batch_size = 50

    # Specify the feature and the label.
    features = ["TRIP_MILES"]
    label = "FARE"

    model_1 = run_experiment(
        training_df, features, label, learning_rate, epochs, batch_size
    )

    training_df.loc[:, "TRIP_MINUTES"] = training_df["TRIP_SECONDS"] / 60

    features = ["TRIP_MILES", "TRIP_MINUTES"]
    label = "FARE"

    model_2 = run_experiment(
        training_df, features, label, learning_rate, epochs, batch_size
    )

    output = predict_fare(model_2, training_df, features, label)
    show_predictions(output)


if __name__ == "__main__":
    main()
