import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def cost_function(x, y, beta):
    return (np.dot((np.dot(x, beta) - y).T, np.dot(x, beta) - y) / y.shape[0]).ravel()


def gradient_descent(x, y, beta, alpha, iterations):
    m = y.shape[0]

    for iteration in range(1, iterations + 1):
        h = np.dot(x, beta)
        loss = h - y
        gradient = np.dot(x.T, loss) / m
        beta = beta - alpha * gradient

        cost = cost_function(x, y, beta)
        print("Cost at iteration " + str(iteration) + " : " + str(cost))

    return beta, cost


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

beta, cost = gradient_descent(X, Y, beta, alpha, iterations)
print("LEAST COST = " + str(cost))
print("BETA at least cost = " + str(beta))
