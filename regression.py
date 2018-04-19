import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_decomposition, svm
from sklearn.linear_model import LinearRegression
import quandl, math

df = quandl.get("WIKI/GOOGL")
print(df.head())

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / (df['Adj. Low'] * 100)
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open']) / (df['Adj. Open'] * 100)

df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]
print(df.head())

forecast_col = 'Adj. Close'
# -99999 acts as outlier; here we train data points with missing values too, as an outlier.
df.fillna(value=-99999, inplace=True)

# forecast range 1% of the total length of duration in data set.
forecast_out = int(math.ceil(0.01 * len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)