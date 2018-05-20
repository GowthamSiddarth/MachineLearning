import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (20.0, 10.0)

data = pd.read_csv('../../data/headbrain.csv')
print(data.head())

X = data.ix[:, 2]
Y = data.ix[:, 3]

X = X.values.reshape(X.shape[0], 1)
Y = Y.values.reshape(Y.shape[0], 1)

model = LinearRegression()
model.fit(X, Y)

y = model.predict(X)

rmse = np.sqrt(mean_squared_error(Y, y))
print("root mean squared error = " + str(rmse))

rse = model.score(X, Y)
print("root squared error = " + str(rse))
