import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def cost_function(x, y, beta):
    return (np.dot((np.dot(x, beta) - y).T, np.dot(x, beta) - y) / y.shape[0]).ravel()


data = pd.read_csv('../../data/student.csv')
print(data.head())

m = data.shape[0]
math = data['Math'].values.reshape((m, 1))
read = data['Reading'].values.reshape((m, 1))
write = data['Writing'].values.reshape((m, 1))

fig = plt.figure()
axes = Axes3D(fig)
axes.scatter(math, read, write, color='#ef1234')
plt.show()

X = np.c_[np.ones(m), math, read]
beta = np.random.rand(3, 1)
Y = write

alpha = 0.01
initial_cost = cost_function(X, Y, beta)
print("INITIAL COST = " + str(initial_cost))
