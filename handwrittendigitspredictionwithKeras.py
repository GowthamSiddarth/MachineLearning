import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Lambda, Flatten
from keras.optimizers import Adam, RMSprop
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split


def standardize(x):
    mean = x.mean().astype(np.float32)
    std = x.std().astype(np.float32)

    return (x - mean) / std


train_data = pd.read_csv('data/hand-written-digit-recognition-train.csv')
test_data = pd.read_csv('data/hand-writtern-digit-recognition-test.csv')

X_train = train_data.ix[:, 1:].values.astype('float32')
y_train = train_data.ix[:, :1].values.astype('int32')

X_test = test_data.values.astype('float32')

X_train = X_train.reshape(X_train.shape[0], 28, 28)
X_test = X_test.reshape(X_test.shape[0], 28, 28)

for i in range(9):
    plt.subplot(330 + (i + 1))
    plt.imshow(X_train[i], cmap='gray')
    plt.title(y_train[i])

plt.show()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

y_train = to_categorical(y_train)
num_classes = y_train.shape[1]
print(num_classes)
