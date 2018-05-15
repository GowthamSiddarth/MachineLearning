import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('data/melb_data.csv')
features_labels = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']

X = data[features_labels]
y = data.Price

train_X, test_X, train_y, test_y = train_test_split(X, y)

pipeline = make_pipeline(Imputer(), RandomForestRegressor())
pipeline.fit(train_X, train_y)

predictions = pipeline.predict(test_X)
mae = mean_absolute_error(test_y, predictions)
print("MEAN ABSOLUTE ERROR: " + str(mae))
