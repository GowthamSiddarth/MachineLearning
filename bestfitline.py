import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style


def best_fit_slope_and_intercept(xs, ys):
    xs_mean, ys_mean = np.mean(xs), np.mean(ys),
    m = ((xs_mean * ys_mean) - np.mean(np.multiply(xs, ys))) / (np.power(xs_mean, 2) - np.mean(np.multiply(xs, xs)))
    b = ys_mean - m * xs_mean

    return m, b


xs = np.array([1, 2, 3, 4, 5], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6], dtype=np.float64)

m, b = best_fit_slope_and_intercept(xs, ys)
print(m, b)

regression_line = [m * x + b for x in xs]

predict_x = 7
predict_y = m * predict_x + b

style.use('ggplot')
plt.scatter(xs, ys, color='#003F72', label='Data')
plt.scatter(predict_x, predict_y, color='#00723F', label='Prediction')
plt.plot(xs, regression_line, label='regression line')
plt.legend(loc=4)
plt.show()
