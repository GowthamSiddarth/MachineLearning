import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['figure.figsize'] = (20.0, 10.0)

data = pd.read_csv('../../data/student.csv')
print(data.head())
