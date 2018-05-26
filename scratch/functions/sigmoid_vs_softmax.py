import numpy as np
from matplotlib import pyplot as plt


def sigmoid(x): return 1 / (1 + np.exp(-x))


def softmax(x): return np.exp(x) / np.sum(np.exp(x))


X = np.arange(0, 20, 0.5)

Y = sigmoid(X)
plt.figure()
plt.plot(X, Y)
plt.xlabel('X')
plt.ylabel('sigmoid(x)')

Y = softmax(X)
plt.figure()
plt.plot(X, Y)
plt.xlabel('X')
plt.ylabel('softmax(x)')

plt.show()
