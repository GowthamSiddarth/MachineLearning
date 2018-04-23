import numpy as np
from matplotlib import pyplot as plt, style
from bestfitline import best_fit_slope_and_intercept


def squared_error(y_orig, y_line): return np.sum(np.power(y_orig - y_line, 2))


def coefficient_of_determination(y_orig, y_line):
    y_mean = np.repeat(np.mean(y_orig), len(y_orig))
    squared_error_regr = squared_error(y_orig, y_line)
    squared_error_mean = squared_error(y_orig, y_mean)

    return 1 - (squared_error_regr / squared_error_mean)


xs = np.array([1, 2, 3, 4, 5], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6], dtype=np.float64)

m, b = best_fit_slope_and_intercept(xs, ys)
regression_line = [(m * x + b) for x in xs]

r_squared_error = coefficient_of_determination(ys, regression_line)
print(r_squared_error)

style.use('ggplot')
plt.scatter(xs, ys, color='#003F72', label='data')
plt.plot(xs, regression_line, label='regression line')
plt.legend(loc=4)
plt.show()
