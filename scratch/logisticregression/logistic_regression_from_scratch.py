import pandas as pd
import numpy as np
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt


def encode_labels(data_frame, labels):
    label_encoder = LabelEncoder()
    for label in labels:
        label_encoder.fit(data_frame[label])
        data_frame[label] = label_encoder.transform(data_frame[label])

    return data_frame


def normalize_features(x_train):
    for feature in x_train.columns:
        x_train[feature] = (x_train[feature] - np.mean(x_train[feature])) / np.std(x_train[feature])

    return x_train


train_data = pd.read_csv('../../data/titanic-train.csv')
test_data = pd.read_csv('../../data/titanic-test.csv')

print(train_data.head())
print(train_data.shape)

plt.figure()
plt.suptitle("Proportion of male & female who survived")
sb.countplot(x="Sex", hue="Survived", data=train_data)
plt.show()

plt.figure()
plt.suptitle("Proportion of economic categories who survived")
sb.countplot(x="Pclass", hue="Survived", data=train_data)
plt.show()
