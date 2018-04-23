import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('data/breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], axis=1, inplace=True)

X = np.array(df.drop(labels=['class'], axis=1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = neighbors.KNeighborsClassifier()
clf.fit(X=X_train, y=y_train)
accuracy = clf.score(X=X_test, y=y_test)
print(accuracy)

examples = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 7, 1, 10, 8, 2, 1]])
examples = examples.reshape(len(examples), -1)
prediction = clf.predict(examples)
print(prediction)
