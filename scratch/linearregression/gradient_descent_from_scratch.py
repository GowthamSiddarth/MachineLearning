import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('../../data/student.csv')
print(data.head())

math = data['Math'].values
read = data['Reading'].values
write = data['Writing'].values

fig = plt.figure()
axes = Axes3D(fig)
axes.scatter(math, read, write, color='#ef1234')
plt.show()
