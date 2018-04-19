from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import quandl, math, os, pickle

linear_classifier_filename = 'linearregression.pickle'
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
print("Prediction for number of days ahead = " + str(forecast_out))

df['label'] = df[forecast_col].shift(-forecast_out)
print(df.tail(n=50))

# drop label values along the column: axis = 1
X = np.array(df.drop(['label'], axis=1))

# scale values between -1 and 1
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df_nas = df[df.isna().any(axis=1)]
# df.dropna(inplace=True)

y = np.array(df['label'])
y = y[:-forecast_out]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

'''
clf = svm.SVR(kernel='linear')
clf.fit(X=X_train, y=y_train)
confidence = clf.score(X=X_test, y=y_test)
print(confidence)
'''

if os.path.exists(linear_classifier_filename):
    # Load the already saved classifier from the file
    pickle_in = open(linear_classifier_filename, 'rb')
    clf = pickle.load(pickle_in)
else:
    # n_jobs is the number of threads to work on for the classifier
    clf = LinearRegression(n_jobs=-1)
    clf.fit(X=X_train, y=y_train)
    confidence = clf.score(X=X_test, y=y_test)
    print(confidence)

    # Store classifier into a file for first time
    with open(linear_classifier_filename, 'wb') as pickle_out:
        pickle.dump(clf, pickle_out)

forecast_set = clf.predict(X=X_lately)

'''
for k in ['linear', 'poly', 'rbf', 'sigmoid']:
    clf = svm.SVR(kernel=k)
    clf.fit(X=X_train, y=y_train)
    confidence = clf.score(X=X_test, y=y_test)
    print(k, confidence)
'''

style.use('ggplot')

# Label : X :: Forecast : X_lately
df['Forecast'] = np.nan

for i in range(len(forecast_set)):
    df.at[df_nas.iloc[i].name, 'Forecast'] = forecast_set[i]

# Plot 'Adj. Close' values from X only
df.iloc[:-forecast_out, df.columns.get_loc('Adj. Close')].plot()
# Plot 'Forecast' values from X_lately only
df['Forecast'].plot()

plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
