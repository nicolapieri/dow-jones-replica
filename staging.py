import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# setting time interval
start_date, end_date = "1993-01-01", "2022-12-31"

# getting Dow Jones index timeseries
DJI = pd.DataFrame(yf.download("^DJI", start=start_date, end=end_date))

# declaring Dow Jones compositions (https://en.wikipedia.org/wiki/Historical_components_of_the_Dow_Jones_Industrial_Average)
# excluded CRM, DOW, GS and V for missing values
DJI_components = ["MMM", "AXP", "AMGN", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO",
                  "HD", "HON", "INTC", "IBM", "JNJ", "JPM", "MCD", "MRK", "MSFT",
                  "NKE", "PG", "TRV", "UNH", "VZ", "WBA", "WMT", "DIS"]

# creating closes and returns dataframes
Closes = pd.DataFrame()
Returns = pd.DataFrame()
for symbol in DJI_components:
    stock = pd.DataFrame(yf.download(symbol, start=start_date, end=end_date))
    Closes[symbol] = stock.loc[:, 'Adj Close']
    Returns[symbol] = (stock.loc[:, 'Adj Close'] - stock.loc[:, 'Adj Close'].shift(1)) / stock.loc[:,
                                                                                         'Adj Close'].shift(1)
Closes.drop(index=Returns.index[0], axis=0, inplace=True)
Closes['sel26'] = Closes.sum(axis=1)
Closes['^DJI'] = DJI['Adj Close']
Returns.drop(index=Returns.index[0], axis=0, inplace=True)
Returns['sel26'] = Returns.mean(axis=1)
Returns['^DJI'] = (DJI['Adj Close'] - DJI['Adj Close'].shift(1)) / DJI['Adj Close'].shift(1)

# train, val, test split
split = Returns.drop('sel26', axis=1)
train = split[(split.index >= pd.to_datetime("1993-01-01")) & (split.index <= pd.to_datetime("2021-12-31"))]
val = split[(split.index >= pd.to_datetime("2022-01-01")) & (split.index <= pd.to_datetime("2022-11-30"))]
test = split[(split.index >= pd.to_datetime("2022-12-01")) & (split.index <= pd.to_datetime("2022-12-31"))]
trainX = train.drop('^DJI', axis=1)
trainY = pd.DataFrame({'^DJI': train['^DJI']})
valX = val.drop('^DJI', axis=1)
valY = pd.DataFrame({'^DJI': val['^DJI']})
testX = test.drop('^DJI', axis=1)
testY = pd.DataFrame({'^DJI': test['^DJI']})


# performance evaluation function
def evaluate(df, portfolio_col, is_test):
    index_hpr = (Returns['^DJI'][-1] - Returns['^DJI'][0]) / Returns['^DJI'][0]

    if is_test:
        benchmark_hpr = (Returns['sel26'][-1] - Returns['sel26'][0]) / Returns['sel26'][0]
        benchmark_active_return = benchmark_hpr - index_hpr
        benchmark_tracking_error = np.std(Returns['sel26'] - Returns['^DJI'])
        info_ratio = benchmark_active_return / benchmark_tracking_error
        print("\nBenchmark (sel26)")
        print("*" * 30)
        print("Active Return:", round(benchmark_active_return, 5))
        print("Tracking Error:", round(benchmark_tracking_error * 10000), " bps")
        print("Information Ratio:", round(info_ratio, 5))
        print("Returns RMSE:", mean_squared_error(Returns['sel26'], Returns['^DJI'], squared=False))

    if portfolio_col:
        portfolio_hpr = (df[portfolio_col][-1] - df[portfolio_col][0]) / df[portfolio_col][0]
        portfolio_active_return = portfolio_hpr - index_hpr
        portfolio_tracking_error = np.std(df[portfolio_col] - Returns['^DJI'])
        info_ratio = portfolio_active_return / portfolio_tracking_error
        print("\nTrial Portfolio")
        print("*" * 30)
        print("Active Return:", round(portfolio_active_return, 5))
        print("Tracking Error:", round(portfolio_tracking_error * 10000), "bps")
        print("Information Ratio:", round(info_ratio, 5))
        print("Returns RMSE:", mean_squared_error(df[portfolio_col], df['^DJI'], squared=False))

    df.plot()
    plt.title("Non-negative Least Squares (NNLS) Optimization")
    plt.show()
    pass
