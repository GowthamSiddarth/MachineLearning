import random
import numpy as np
from matplotlib import pyplot as plt, style
from bestfitline import best_fit_slope_and_intercept
from rsquared import coefficient_of_determination


def create_data_set(hm, variance, step=2, correlation=False):
    val, ys = 1, []
    for _ in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step

    xs = [_ for _ in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


xs, ys = create_data_set(40, 40, correlation='pos')
m, b = best_fit_slope_and_intercept(xs, ys)
regression_line = [(m * x + b) for x in xs]
r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

style.use('ggplot')
plt.scatter(xs, ys, color='#003F72', label='data')
plt.plot(xs, regression_line, label='regression line')
plt.legend(loc=4)
plt.show()
