import pandas as pd


def get_train_and_test_data():
    train_data = pd.read_csv('data/melbourne-house-prices-train-data.csv')
    test_data = pd.read_csv('data/melbourne-house-prices-test-data.csv')

    return train_data, test_data


def drop_columns_with_missing_values(train_data, test_data):
    cols_with_missing_values = [col for col in train_data.columns if train_data[col].isnull().any()]

    reduced_train_data = train_data.drop(cols_with_missing_values, axis=1)
    reduced_test_data = test_data.drop(cols_with_missing_values, axis=1)

    return reduced_train_data, reduced_test_data


if __name__ == '__main__':
    train_data, test_data = get_train_and_test_data()

    print(train_data.describe())
    print(test_data.describe())

    reduced_train_data, reduced_test_data = drop_columns_with_missing_values(train_data, test_data)

    print(reduced_train_data.describe())
    print(reduced_test_data.describe())
