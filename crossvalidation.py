import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import Imputer
from sklearn.pipeline import make_pipeline

data = pd.read_csv('data/melb_data.csv')
feature_labels = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']

y = data.Price
X = data[feature_labels]

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25)

pipeline = make_pipeline(Imputer(), RandomForestRegressor())
scores = cross_val_score(pipeline, train_X, train_y, scoring='neg_mean_absolute_error')

print(-1 * scores.mean())
