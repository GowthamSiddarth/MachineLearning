import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt

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
