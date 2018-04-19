from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np
import datetime
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import quandl, math

df = quandl.get("WIKI/GOOGL")
# print(df.head())

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / (df['Adj. Low'] * 100)
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open']) / (df['Adj. Open'] * 100)

df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]
print(df.tail())

forecast_col = 'Adj. Close'
# -99999 acts as outlier; here we train data points with missing values too, as an outlier.
df.fillna(value=-99999, inplace=True)

# forecast range 1% of the total length of duration in data set.
forecast_out = int(math.ceil(0.01 * len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
print(df.tail(n=50))

# drop label values along the column: axis = 1
X = np.array(df.drop(['label'], axis=1))

# scale values between -1 and 1
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)

y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

'''
clf = svm.SVR()
clf.fit(X=X_train, y=y_train)
confidence = clf.score(X=X_test, y=y_test)
print(confidence)
'''

# n_jobs is the number of threads to work on for the classifier
clf = LinearRegression(n_jobs=-1)
clf.fit(X=X_train, y=y_train)
confidence = clf.score(X=X_test, y=y_test)
print(confidence)

forecast_set = clf.predict(X=X_lately)

'''
for k in ['linear', 'poly', 'rbf', 'sigmoid']:
    clf = svm.SVR(kernel=k)
    clf.fit(X=X_train, y=y_train)
    confidence = clf.score(X=X_test, y=y_test)
    print(k, confidence)
'''

style.use('ggplot')
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date.strftime("%Y-%m-%d")] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

print(df.tail(n=50))
df['Adj. Close'].plot()
df['Forecast'].plot()

plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
