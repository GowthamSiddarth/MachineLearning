import pandas as pd
import numpy as np

data = pd.read_csv('../../data/student.csv')
print(data.head())

m = data.shape[0]
math = data['Math'].values.reshape(m, 1)
read = data['Reading'].values.reshape(m, 1)
write = data['Writing'].values.reshape(m, 1)

math = (math - np.mean(math)) / np.std(math)
read = (read - np.mean(read)) / np.std(read)
write = (write - np.mean(write)) / np.std(write)

X = np.c_[math, read]
Y = write
