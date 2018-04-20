import numpy as np


def best_fit_slope(xs, ys):
    return ((np.mean(xs) * np.mean(ys)) - np.mean(np.multiply(xs, ys))) / (
        np.power(np.mean(xs), 2) - np.mean(np.multiply(xs, xs)))


xs = np.asarray([1, 2, 3, 4, 5])
ys = np.array([5, 4, 6, 5, 6])

m = best_fit_slope(xs=xs, ys=ys)
print(m)
