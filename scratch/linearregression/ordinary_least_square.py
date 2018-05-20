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
        x = x.values.reshape(x.shape[0], 1)
    if 1 == y.ndim:
        y = y.values.reshape(y.shape[0], 1)

    m = y.shape[0]
    return (np.dot(x.T, y)[0][0] - (np.sum(x) * np.sum(y)) / m).ravel()


def variance(x):
    if 1 == x.ndim:
        x = x.values.reshape(x.shape[0], 1)

    m = x.shape[0]
    return (np.dot(x.T, x) - (np.sum(x) ** 2) / m).ravel()


c, v = covariance(X, y), variance(X)
print("covariance = " + str(c))
print("variance = " + str(v))

beta1 = c[0] / v[0]
beta0 = np.mean(y) - beta1 * np.mean(X)

print(beta0, beta1)
