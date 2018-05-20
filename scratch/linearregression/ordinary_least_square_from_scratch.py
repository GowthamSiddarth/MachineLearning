import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def covariance(x, y):
    m = y.shape[0]
    return (np.dot(x.T, y)[0][0] - (np.sum(x) * np.sum(y)) / m).ravel()


def variance(x):
    m = x.shape[0]
    return (np.dot(x.T, x) - (np.sum(x) ** 2) / m).ravel()


def r_squared_error(y_orig, y_pred):
    return (1 - np.dot((y_orig - y_pred).T, y_orig - y_pred) \
            / np.dot((y_orig - np.mean(y_orig)).T, y_orig - np.mean(y_orig))).ravel()


def root_mean_square_error(y_orig, y_pred):
    return np.sqrt(np.dot((y_orig - y_pred).T, y_orig - y_pred)
                   / y_orig.shape[0]).ravel()


plt.rcParams['figure.figsize'] = (20.0, 10.0)

data = pd.read_csv('../../data/headbrain.csv')
print(data.head())

X = data.ix[:, 2]
Y = data.ix[:, 3]

X = X.values.reshape(X.shape[0], 1)
Y = Y.values.reshape(Y.shape[0], 1)

c, v = covariance(X, Y), variance(X)
print("covariance = " + str(c))
print("variance = " + str(v))

beta1 = c[0] / v[0]
beta0 = np.mean(Y) - beta1 * np.mean(X)
print("beta0 = " + str(beta0))
print("beta1 = " + str(beta1))

y_pred = beta0 + beta1 * X

rmse = root_mean_square_error(Y, y_pred)
print("root mean square error = " + str(rmse))

rse = r_squared_error(Y, y_pred)
print("root squared error = " + str(rse))

max_x = np.max(X) + 100
min_x = np.min(X) - 100

x_plot = np.linspace(min_x, max_x, 1000)
y_plot = beta0 + beta1 * x_plot

plt.plot(x_plot, y_plot, color='#58b970', label='Regression Line')
plt.scatter(X, Y, c='#ef5423', label='Scatter Plot')
plt.legend()
plt.show()
