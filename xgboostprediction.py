import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

melb_data = pd.read_csv('data/melbourne-house-prices-train-data.csv')
melb_data.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = melb_data.SalePrice
X = melb_data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])

train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)

imputer = Imputer()
train_X = imputer.fit_transform(train_X)
test_X = imputer.transform(test_X)

model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
model.fit(train_X, train_y, early_stopping_rounds=5, eval_set=[(test_X, test_y)], verbose=False)

predictions = model.predict(test_X)
mae = mean_absolute_error(test_y, predictions)
print("MEAN ABSOLUTE ERROR: " + str(mae))
