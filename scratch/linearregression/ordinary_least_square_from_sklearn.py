import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (20.0, 10.0)

data = pd.read_csv('../../data/headbrain.csv')
print(data.head())

X = data.ix[:, 2]
Y = data.ix[:, 3]