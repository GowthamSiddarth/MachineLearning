import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from matplotlib import pyplot as plt

partial_dependence_features_labels = ['Age', 'Fare']


def get_data():
    data_file_path = 'data/titanic-train.csv'
    data = pd.read_csv(data_file_path)
    y = data.Survived
    X = data[partial_dependence_features_labels]
    X = pd.get_dummies(X)

    from sklearn.preprocessing import Imputer
    imputer = Imputer()
    X = imputer.fit_transform(X)

    return X, y


X, y = get_data()
model = GradientBoostingRegressor()
model.fit(X, y)
fig, axes = plot_partial_dependence(model, feature_names=partial_dependence_features_labels, features=[0, 1], X=X,
                                    grid_resolution=10)

plt.show()
