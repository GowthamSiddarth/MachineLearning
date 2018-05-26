import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


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
    features = data_set[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]
    normalize_features(features)

    if data_set_type == 'train':
        y_train = data_set['Survived']
        y_train = y_train.reshape(y_train.shape[0], 1)
        return features.as_matrix(), y_train
    else:
        passenger_id = data_set['PassengerId']
        return features.as_matrix(), passenger_id

    return x


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


train_X, train_y = get_preprocessed_data('train')
test_X, passenger_id = get_preprocessed_data('test')

alpha, weights, bias, m = 0.01, np.random.normal(0, 0.1, 9), np.random.normal(0, 0.1, 9), train_y.shape[0]
print(train_X.shape)
print(train_y.shape)


