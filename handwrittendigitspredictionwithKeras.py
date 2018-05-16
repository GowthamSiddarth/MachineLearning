import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Lambda, Flatten
from keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('data/hand-written-digit-recognition-train.csv')
test_data = pd.read_csv('data/hand-writtern-digit-recognition-test.csv')

X_train = train_data.ix[:, 1:].values.as_type('float32')
y_train = train_data.ix[:, :1].values.as_type('int32')

X_test = test_data.values.as_type('float32')
