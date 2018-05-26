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
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    features_normalized = normalize_features(data_set, features)

    if data_set_type == 'train':
        y_train = data_set['Survived']
        y_train = y_train.values.reshape(y_train.shape[0], 1)
        return features_normalized.as_matrix(), y_train
    else:
        passenger_id = data_set['PassengerId']
        return features_normalized.as_matrix(), passenger_id

    return x


def encode_labels(data_frame, labels):
    label_encoder = LabelEncoder()
    for label in labels:
        label_encoder.fit(data_frame[label])
        data_frame[label] = label_encoder.transform(data_frame[label])

    return data_frame


def normalize_features(data_set, features):
    for feature in features:
        data_set.loc[:, feature] = (data_set[feature] - np.mean(data_set[feature])) / np.std(data_set[feature])

    return data_set[features]


def sigmoid(x): return 1 / (1 + np.exp(-x))


train_X, train_y = get_preprocessed_data('train')
test_X, passenger_id = get_preprocessed_data('test')

m, epochs = train_y.shape[0], 10000
alpha, weights, bias = 0.01, np.random.normal(0, 0.1, 9).reshape(9, 1), np.random.normal(0, 0.1, m).reshape(m, 1)
print(train_X.shape)
print(train_y.shape)

for epoch in range(1, epochs + 1):
    Z = np.dot(train_X, weights) + bias
    A = sigmoid(Z)

    loss = np.sum(-(np.dot(train_y.T, np.log(A)) + np.dot(1 - train_y.T, np.log(1 - A)))) / m
    if 0 == epoch % 100:
        print("Cost at iteration " + str(epoch) + " is: " + str(loss))

    dZ = A - train_y
    dw = np.dot(train_X.T, dZ) / m
    db = np.sum(dZ) / m

    weights = weights - alpha * dw
    bias = bias - alpha * db
