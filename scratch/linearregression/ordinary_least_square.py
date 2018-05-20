import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (20.0, 10.0)

data = pd.read_csv('../../data/headbrain.csv')
print(data.shape)
print(data.head())

X = data.ix[:, 2]
y = data.ix[:, 3]

print(X.shape)
print(y.shape)


def covariance(x, y):
    if 1 == x.ndim:
        x = x.reshape(x.shape[0], 1)
    if 1 == y.ndim:
        y = y.reshape(y.shape[0], 1)

    m = y.shape[0]
    return x * y - (np.sum(x) * np.sum(y)) / m


def variance(x):
    if 1 == x.ndim:
        x = x.reshape(x.shape[0], 1)

    m = x.shape[0]
    return x.T * x - (np.sum(x) ** 2) / m

