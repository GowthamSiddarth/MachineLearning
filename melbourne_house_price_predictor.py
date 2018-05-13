import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

train_data_file = 'data/melbourne-house-prices-train-data.csv'
data = pd.read_csv(train_data_file)
print(data.describe())

columns = data.columns
print(columns)

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X, y = data[features], data.SalePrice
print(X.describe())

model = DecisionTreeRegressor()
model.fit(X, y)

print("Predictions:")
predictions = model.predict(X)
print(predictions)

mae = mean_absolute_error(predictions, y)
print("MEAN ABSOLUTE ERROR: " + str(mae))
