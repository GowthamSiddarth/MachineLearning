import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def cost_function(x, y, beta):
    return (np.dot((np.dot(x, beta) - y).T, np.dot(x, beta) - y) / y.shape[0]).ravel()


def gradient_descent(x, y, beta, alpha, iterations):
    cost_history = [0] * iterations
    m = y.shape[0]

    for iteration in range(1, iterations + 1):
        h = np.dot(x, beta)
        loss = h - y
        gradient = np.dot(x.T, loss) / m
        beta = beta - alpha * gradient

        cost = cost_function(x, y, beta)
        cost_history[iteration - 1] = cost
        print("Cost at iteration " + str(iteration) + " : " + str(cost))

    return beta, cost_history


def root_mean_square_error(y_pred, y_orig):
    return np.sqrt(np.dot((y_pred - y_orig).T, y_pred - y_orig) / y_orig.shape[0]).ravel()


def root_squared_error(y_pred, y_orig):
    return 1 - np.dot((y_pred - y_orig).T, y_pred - y_orig) \
               / np.dot((y_orig - np.mean(y_orig)).T, y_orig - np.mean(y_orig))


data = pd.read_csv('../../data/student.csv')
print(data.head())

m = data.shape[0]
math = data['Math'].values.reshape((m, 1))
read = data['Reading'].values.reshape((m, 1))
write = data['Writing'].values.reshape((m, 1))

math = (math - np.mean(math)) / np.std(math)
read = (read - np.mean(read)) / np.std(read)
write = (write - np.mean(write)) / np.std(write)

fig = plt.figure()
axes = Axes3D(fig)
axes.scatter(math, read, write, color='#ef1234')
plt.show()

X = np.c_[np.ones(m), math, read]
beta = np.random.rand(3, 1)
print("INITIAL BETA = " + str(beta))
Y = write

alpha, iterations = 0.01, 4000
initial_cost = cost_function(X, Y, beta)
print("INITIAL COST = " + str(initial_cost))

beta, cost_history = gradient_descent(X, Y, beta, alpha, iterations)
print("LEAST COST = " + str(cost_history[-1]))
print("BETA at least cost = " + str(beta))

plt.plot(np.linspace(1, iterations, num=4000), cost_history)
plt.xlabel("Number of Iterations")
plt.ylabel("Cost")
plt.show()

y_pred = np.dot(X, beta)

rmse = root_mean_square_error(y_pred, Y)
print("ROOT MEAN SQUARE ERROR = " + str(rmse))

rse = root_squared_error(y_pred, Y)
print("ROOT_SQUARED_ERROR = " + str(rse))
