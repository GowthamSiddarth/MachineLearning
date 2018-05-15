import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

train_data = pd.read_csv('data/melbourne-house-prices-train-data.csv')
test_data = pd.read_csv('data/melbourne-house-prices-test-data.csv')

train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)

target = train_data.SalePrice

cols_with_missing_values = [col for col in train_data.columns if train_data[col].isnull().any()]
candidate_train_data = train_data.drop(['Id', 'SalePrice'] + cols_with_missing_values, axis=1)
candidate_test_data = test_data.drop(['Id'] + cols_with_missing_values, axis=1)

low_cardinal_cols = [col for col in candidate_train_data.columns
                     if candidate_train_data[col].dtype == 'object' and
                     candidate_train_data[col].nunique() < 10]
numeric_cols = [col for col in candidate_train_data.columns
                if candidate_train_data[col].dtype in ['int64', 'float64']]

features = low_cardinal_cols + numeric_cols

features_from_train_data = candidate_train_data[features]
features_from_test_data = candidate_test_data[features]

print(features_from_train_data.dtypes.sample(10))


def get_mae(X, y):
    return -1 * cross_val_score(RandomForestRegressor(50), X, y, scoring='neg_mean_absolute_error').mean()


one_hot_encoded_features_from_train_data = pd.get_dummies(features_from_train_data)
features_without_categorical_values = features_from_train_data.select_dtypes(exclude=['object'])

mae_without_encoding = get_mae(features_without_categorical_values, target)
mae_with_encoding = get_mae(one_hot_encoded_features_from_train_data, target)

print(mae_without_encoding)
print(mae_with_encoding)
