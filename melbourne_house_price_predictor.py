import pandas as pd

train_data_file = 'data/melbourne-house-prices-train-data.csv'
data = pd.read_csv(train_data_file)

print(data.describe())

columns = data.columns
print(columns)

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = data[features]

print(X.describe())

y = data.SalePrice
