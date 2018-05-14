import pandas as pd

melb_data = pd.read_csv('data/melb_data.csv')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def score_dataset(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return mean_absolute_error(y_test, predictions)


melb_target = melb_data.Price
melb_predictors = melb_data.drop(['Price'], axis=1)

melb_numeric_predictors = melb_predictors.select_dtypes(exclude=['object'])

train_X, test_X, train_y, test_y = train_test_split(melb_numeric_predictors, melb_target,
                                                    train_size=0.7, test_size=0.3, random_state=0)
