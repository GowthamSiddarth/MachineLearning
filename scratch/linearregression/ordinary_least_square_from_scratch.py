import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (20.0, 10.0)

data = pd.read_csv('../../data/headbrain.csv')
print(data.head())

X = data.ix[:, 2]
Y = data.ix[:, 3]


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


def r_squared_error(y_orig, y_pred):
    return 1 - np.dot((y_orig - y_pred).T, y_orig - y_pred) \
               / np.dot((y_orig - np.mean(y_orig)).T, y_orig - np.mean(y_orig))


def root_mean_square_error(y_orig, y_pred):
    return np.sqrt(np.dot(y_orig - y_pred, (y_orig - y_pred).T)
                   / y_orig.shape[0])


c, v = covariance(X, Y), variance(X)
print("covariance = " + str(c))
print("variance = " + str(v))

beta1 = c[0] / v[0]
beta0 = np.mean(Y) - beta1 * np.mean(X)
print(beta0, beta1)

y_pred = beta0 + beta1 * X

rmse = root_mean_square_error(Y, y_pred)
print(rmse)

rse = r_squared_error(Y, y_pred)
print(rse)

max_x = np.max(X) + 100
min_x = np.min(X) - 100

x_plot = np.linspace(min_x, max_x, 1000)
y_plot = beta0 + beta1 * x_plot

plt.plot(x_plot, y_plot, color='#58b970', label='Regression Line')
plt.scatter(X, Y, c='#ef5423', label='Scatter Plot')
plt.legend()
plt.show()
