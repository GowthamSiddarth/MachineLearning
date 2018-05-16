import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def view_image(data_set, loc):
    img = data_set.iloc[loc, 1:].as_matrix()
    img = img.reshape((28, 28))

    fig1 = plt.figure(1)
    plt.imshow(img, cmap='gray')
    plt.title(data_set.iloc[loc, 0])

    fig2 = plt.figure(2)
    plt.hist(data_set.iloc[loc, 1:])
    
    plt.show()


train_data = pd.read_csv('data/hand-written-digit-recognition-train.csv')
test_data = pd.read_csv('data/hand-writtern-digit-recognition-test.csv')

X, y = train_data.iloc[:5000, 1:], train_data.iloc[:5000, :1]
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)

view_image(train_data, 9)
