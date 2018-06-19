import numpy as np
import pandas as pd


def normalize(x): return (x - np.mean(x)) / np.std(x)


# ../../data/hand-written-digit-recognition-train.csv
def get_data(path, instances, normalization=True):
    train_data = pd.read_csv(path)
    train_X, train_Y = train_data.iloc[:instances, 1:], train_data.iloc[:instances, :1]

    if normalization:
        train_X = normalize(train_X)

    return train_X, train_Y


if __name__ == "__main__":
    print("Starting...")
    train_data_path = "../../data/hand-written-digit-recognition-train.csv"
    instances = 10000

    train_X, train_Y = get_data(train_data_path, instances)
    print(train_X.shape)
    print(train_Y.shape)
