import numpy as np
from sklearn import svm, preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('data/breast-cancer-wisconsin.data.txt')
df.replace(to_replace='?', value=-99999, inplace=True)
df.drop(labels=['id'], axis=1, inplace=True)

X = np.array(df.drop(['class'], axis=1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = svm.SVC()
clf.fit(X=X_train, y=y_train)
confidence = clf.score(X=X_test, y=y_test)
print("confidence = " + str(confidence))

example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1]])
example_measures = example_measures.reshape((len(example_measures), -1))
prediction = clf.predict(example_measures)
print("prediction = " + str(prediction))
