import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

train_data_file = 'data/melbourne-house-prices-train-data.csv'
data = pd.read_csv(train_data_file)
print(data.describe())

columns = data.columns
print(columns)

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X, y = data[features], data.SalePrice
print(X.describe())

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=0)

model = DecisionTreeRegressor()
model.fit(train_X, train_y)

print("Predictions:")
predictions = model.predict(test_X)
print(predictions)

mae = mean_absolute_error(predictions, test_y)
print("MEAN ABSOLUTE ERROR: " + str(mae))
