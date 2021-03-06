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


def impute_missing_data(X_train, X_test):
    from sklearn.preprocessing import Imputer

    imputer = Imputer()
    imputed_X_train = imputer.fit_transform(X_train)
    imputed_X_test = imputer.transform(X_test)

    return imputed_X_train, imputed_X_test


def impute_missing_data_with_missing_columns(X_train, X_test):
    from sklearn.preprocessing import Imputer

    imputed_X_train_plus = X_train.copy()
    imputed_X_test_plus = X_test.copy()

    cols_with_missing_values = [col for col in X_train.columns if X_train[col].isnull().any()]
    for col in cols_with_missing_values:
        imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
        imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()

    imputer = Imputer()
    imputed_X_train_plus = imputer.fit_transform(imputed_X_train_plus)
    imputed_X_test_plus = imputer.transform(imputed_X_test_plus)

    return imputed_X_train_plus, imputed_X_test_plus


melb_target = melb_data.Price
melb_predictors = melb_data.drop(['Price'], axis=1)

melb_numeric_predictors = melb_predictors.select_dtypes(exclude=['object'])

train_X, test_X, train_y, test_y = train_test_split(melb_numeric_predictors, melb_target,
                                                    train_size=0.7, test_size=0.3, random_state=0)

reduced_train_X, reduced_test_X = drop_missing_columns(train_X, test_X)
score_with_drop_columns = score_dataset(reduced_train_X, train_y, reduced_test_X, test_y)

print(score_with_drop_columns)

imputed_train_X, imputed_test_X = impute_missing_data(train_X, test_X)
score_with_imputed_data = score_dataset(imputed_train_X, train_y, imputed_test_X, test_y)

print(score_with_imputed_data)

imputed_train_X_plus, imputed_test_X_plus = impute_missing_data(train_X, test_X)
score_with_imputed_data_plus = score_dataset(imputed_train_X_plus, train_y, imputed_test_X_plus, test_y)

print(score_with_imputed_data_plus)
