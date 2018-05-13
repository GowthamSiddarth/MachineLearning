from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd

train_data_file = 'data/melbourne-house-prices-train-data.csv'
train_data = pd.read_csv(train_data_file)
print(train_data.describe())

columns = train_data.columns
print(columns)

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X, y = train_data[features], train_data.SalePrice
print(X.describe())

train_X, dev_X, train_y, dev_y = train_test_split(X, y, random_state=0)

model = RandomForestRegressor()
model.fit(train_X, train_y)

dev_predictions = model.predict(dev_X)
dev_mae = mean_absolute_error(dev_y, dev_predictions)
print("MEAN ABSOLUTE ERROR: " + str(dev_mae))

test_data_file = 'data/melbourne-house-prices-test-data.csv'
test_data = pd.read_csv(test_data_file)

test_X = test_data[features]
predictions = model.predict(test_X)
print(predictions)

my_predictions = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predictions})
my_predictions.to_csv('data/melbourne-house-prices-predictions.csv', index=False)
