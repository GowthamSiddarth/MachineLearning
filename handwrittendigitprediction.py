import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from matplotlib import pyplot as plt


def view_image(img, label):
    plt.figure(1)
    plt.hist(img)

    plt.figure(2)
    img = img.as_matrix()
    img = img.reshape((28, 28))
    plt.imshow(img, cmap='binary')
    plt.title(label)

    plt.show()


train_data = pd.read_csv('data/hand-written-digit-recognition-train.csv')
test_data = pd.read_csv('data/hand-writtern-digit-recognition-test.csv')

X, y = train_data.iloc[:5000, 1:], train_data.iloc[:5000, :1]
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)

train_X[train_X > 0] = 1
test_X[test_X > 0] = 1

view_image(train_X.iloc[9], train_y.iloc[9])
clf = SVC()
clf.fit(train_X, train_y.values.ravel())
score = clf.score(test_X, test_y)

print(score)
