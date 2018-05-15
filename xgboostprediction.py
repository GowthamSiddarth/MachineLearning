import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

melb_data = pd.read_csv('data/melb_data.csv')
melb_data.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = melb_data.SalePrice
X = melb_data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])

train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)

imputer = Imputer()
train_X = imputer.fit_transform(train_X)
train_y = imputer.transform(test_X)
