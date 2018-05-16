import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Lambda, Flatten
from keras.layers import BatchNormalization, Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split

train_data = pd.read_csv('data/hand-written-digit-recognition-train.csv')
test_data = pd.read_csv('data/hand-writtern-digit-recognition-test.csv')

X_train = train_data.ix[:, 1:].values.astype('float32')
y_train = train_data.ix[:, :1].values.astype('int32')

X_test = test_data.values.astype('float32')

X_train = X_train.reshape(X_train.shape[0], 28, 28)
X_test = X_test.reshape(X_test.shape[0], 28, 28)

for i in range(9):
    plt.subplot(330 + (i + 1))
    plt.imshow(X_train[i], cmap='gray')
    plt.title(y_train[i])

plt.show()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)


def standardize(x): return (x - mean_px) / std_px


y_train = to_categorical(y_train)
num_classes = y_train.shape[1]
print(num_classes)

plt.title(y_train[9])
plt.plot(y_train[9])
plt.xticks(range(10))
plt.show()

seed = 9
np.random.seed(seed)

model = Sequential()
model.add(Lambda(standardize, input_shape=(28, 28, 1)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
gen = image.ImageDataGenerator()

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.1, random_state=42)
train_batches = gen.flow(X_train, y_train, batch_size=64)
val_batches = gen.flow(X_val, y_val, batch_size=64)

history = model.fit_generator(train_batches, train_batches.n, nb_epoch=1, validation_data=val_batches,
                              nb_val_samples=val_batches.n)

history_dict = history.history
train_loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(train_loss_values) + 1)

plt.plot(epochs, train_loss_values, 'bo')
plt.plot(epochs, val_loss_values, 'b+')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

train_acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, train_acc_values, 'bo')
plt.plot(epochs, val_acc_values, 'b+')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

predictions = model.predict_classes(X_test, verbose=0)
submissions = pd.DataFrame({"ImageId": list(range(1, len(predictions) + 1)), "Label": predictions})
submissions.to_csv('data/hand-written-digit-recognition-results-with-keras.csv', index=False, header=True)
