import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def get_mae(max_leaf_nodes, train_X, test_X, train_y, test_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    predictions = model.predict(test_X)
    return mean_absolute_error(test_y, predictions)


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

for max_leaf_nodes in [5, 50, 500, 5000]:
    mae = get_mae(max_leaf_nodes, train_X, test_X, train_y, test_y)
    print("Max leaf nodes: %d   \t\tMean Absolute Error: %d" % (max_leaf_nodes, mae))
