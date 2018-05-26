import pandas as pd
import numpy as np
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt


def get_preprocessed_data(data_set_type):
    if data_set_type == 'train':
        data_set = pd.read_csv('../../data/titanic-train.csv')
    elif data_set_type == 'test':
        data_set = pd.read_csv('../../data/titanic-test.csv')
    else:
        raise Exception('Invalid Argument for data_set_type')

    data_set['Cabin'] = data_set['Cabin'].fillna('C9')
    data_set['Age'] = data_set['Age'].fillna(data_set['Age'].mean())
    data_set['Embarked'] = data_set['Embarked'].fillna('Q')

    data_set = encode_labels(data_set, ['Sex', 'Ticket', 'Cabin', 'Embarked'])
    X_train = data_set[['PClass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]
    Y_train = data_set['Survived']

    normalize_features(X_train)

    return X_train.as_matrix(), Y_train.as_matrix()


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
