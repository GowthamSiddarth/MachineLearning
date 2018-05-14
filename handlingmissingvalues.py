import pandas as pd

melb_data = pd.read_csv('data/melb_data.csv')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

melb_target = melb_data.Price
melb_predictors = melb_data.drop(['Price'], axis=1)

melb_numeric_predictors = melb_predictors.select_dtypes(exclude=['object'])
