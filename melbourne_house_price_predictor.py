import pandas as pd

train_data_file = 'data/melbourne-house-prices-train-data.csv'
data = pd.read_csv(train_data_file)

print(data.describe())
