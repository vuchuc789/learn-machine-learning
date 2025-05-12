import pandas as pd

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format


def main():
    training_df = pd.read_csv(filepath_or_buffer="california_housing_train.csv")
    print(training_df.describe())


if __name__ == "__main__":
    main()
