import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from matplotlib import pyplot as plt

partial_dependence_features_labels = ['Distance', 'Landsize', 'BuildingArea']


def get_data():
    data_file_path = 'data/melb_data.csv'
    data = pd.read_csv(data_file_path)
    y = data.Price
    X = data[partial_dependence_features_labels]

    from sklearn.preprocessing import Imputer
    imputer = Imputer()
    X = imputer.fit_transform(X)

    return X, y


X, y = get_data()
model = GradientBoostingRegressor()
model.fit(X, y)
plots = plot_partial_dependence(model, feature_names=partial_dependence_features_labels, features=[0, 2], X=X,
                                grid_resolution=10)
plt.show()
