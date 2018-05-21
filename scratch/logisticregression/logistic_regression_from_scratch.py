import pandas as pd

train_data = pd.read_csv('../../data/titanic-train.csv')
test_data = pd.read_csv('../../data/titanic-test.csv')

print(train_data.head())
print(train_data.shape)
