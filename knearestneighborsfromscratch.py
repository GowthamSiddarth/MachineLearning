import numpy as np
from matplotlib import pyplot as plt, style
import warnings
from collections import Counter


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


data_set = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}
new_features = [5, 7]

style.use('ggplot')
[[plt.scatter(features[0], features[1], s=100, c=category) for features in data_set[category]] for category in
 data_set]
plt.scatter(new_features[0], new_features[1], s=100)
plt.show()

result = k_nearest_neighbors(data_set, new_features)
print(result)

[[plt.scatter(features[0], features[1], s=100, c=category) for features in data_set[category]] for category in
 data_set]
plt.scatter(new_features[0], new_features[1], s=100, c=result)
plt.show()
