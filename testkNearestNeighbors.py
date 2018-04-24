import random
import pandas as pd
from collections import Counter
import numpy as np
import warnings


def k_nearest_neighbors(data_set, predict, k=3):
    if len(data_set) >= k:
        warnings.warn("")

    distances = []
    for category in data_set:
        for features in data_set[category]:
            euclid_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append((euclid_distance, category))

    votes = [distance[1] for distance in sorted(distances)[:k]]
    return Counter(votes).most_common(1)[0][0]


df = pd.read_csv('data/breast-cancer-wisconsin.data.txt')
df.replace(to_replace='?', value=-99999, inplace=True)
df.drop(labels=['id'], axis=1, inplace=True)

full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

train_set = {2: [], 4: []}
test_set = {2: [], 4: []}

test_size = 0.2
train_data = full_data[:-int(test_size * len(full_data))]
test_data = full_data[-int(test_size * len(full_data)):]

print("train data")
print(train_data)
print("test data")
print(test_data)

for record in train_data:
    train_set[record[-1]].append(record[:-1])

for record in test_data:
    test_set[record[-1]].append(record[:-1])

correct, total = 0, 0

for category in test_set:
    for data_set in test_set[category]:
        vote = k_nearest_neighbors(train_set, data_set, k=5)
        if category == vote:
            correct = correct + 1

        total = total + 1

print("Accuracy = " + str(vote / total))
