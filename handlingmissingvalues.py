import pandas as pd


def get_train_and_test_data():
    train_data = pd.read_csv('data/melbourne-house-prices-train-data.csv')
    test_data = pd.read_csv('data/melbourne-house-prices-test-data.csv')

    return train_data, test_data


if __name__ == '__main__':
    train_data, test_data = get_train_and_test_data()

    print(train_data.describe())
    print(test_data.describe())
