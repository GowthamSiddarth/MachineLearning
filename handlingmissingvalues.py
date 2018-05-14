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


def drop_missing_columns(X_train, X_test):
    cols_with_missing_values = [col for col in X_train.columns if X_train[col].isnull().any()]
    reduced_X_train = X_train.drop(cols_with_missing_values, axis=1)
    reduced_X_test = X_test.drop(cols_with_missing_values, axis=1)

    return reduced_X_train, reduced_X_test


melb_target = melb_data.Price
melb_predictors = melb_data.drop(['Price'], axis=1)

melb_numeric_predictors = melb_predictors.select_dtypes(exclude=['object'])

train_X, test_X, train_y, test_y = train_test_split(melb_numeric_predictors, melb_target,
                                                    train_size=0.7, test_size=0.3, random_state=0)

reduced_train_X, reduced_test_X = drop_missing_columns(train_X, test_X)
score_with_drop_columns = score_dataset(reduced_train_X, train_y, reduced_test_X, test_y)

print(score_with_drop_columns)
