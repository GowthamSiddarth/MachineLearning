import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (20.0, 10.0)

data = pd.read_csv('../../data/headbrain.csv')
print(data.head())

X = data.ix[:, 2]
Y = data.ix[:, 3]

model = LinearRegression()
model.fit(X, Y)

y = model.predict(X)
print(y)
