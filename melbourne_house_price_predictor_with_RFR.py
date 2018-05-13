from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd

train_data_file = 'data/melbourne-house-prices-train-data.csv'
data = pd.read_csv(train_data_file)
print(data.describe())

columns = data.columns
print(columns)

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X, y = data[features], data.SalePrice
print(X.describe())

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=0)

model = RandomForestRegressor()
model.fit(train_X, train_y)

predictions = model.predict(test_X)
mae = mean_absolute_error(test_y, predictions)
print("MEAN ABSOLUTE ERROR: " + str(mae))
