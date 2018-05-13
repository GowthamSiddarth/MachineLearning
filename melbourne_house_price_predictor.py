import pandas as pd

train_data_file = 'data/melbourne-house-prices-train-data.csv'
data = pd.read_csv(train_data_file)

print(data.describe())

columns = data.columns
print(columns)

interested_columns_labels = ['YearBuilt', '1stFlrSF']
interested_columns_data = data[interested_columns_labels]

print(interested_columns_data.describe())
