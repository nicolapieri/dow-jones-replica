import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
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
    yahoodata = pd.DataFrame(yf.download(symbol, start=start_date, end=end_date))
    Closes[symbol] = yahoodata.loc[:, 'Adj Close']
    Returns[symbol] = (yahoodata.loc[:, 'Adj Close'] - yahoodata.loc[:, 'Adj Close'].shift(1)) / yahoodata.loc[:, 'Adj Close'].shift(1)
Closes.drop(index=Returns.index[0], axis=0, inplace=True)
Closes['^DJI'] = DJI['Adj Close']
Returns.drop(index=Returns.index[0], axis=0, inplace=True)
Returns['^DJI'] = (DJI['Adj Close'] - DJI['Adj Close'].shift(1)) / DJI['Adj Close'].shift(1)

# train, val, test sets split
train = Returns[(Returns.index >= pd.to_datetime("1993-01-01")) & (Returns.index <= pd.to_datetime("2021-12-31"))]
val = Returns[(Returns.index >= pd.to_datetime("2022-01-01")) & (Returns.index <= pd.to_datetime("2022-11-30"))]
test = Returns[(Returns.index >= pd.to_datetime("2022-12-01")) & (Returns.index <= pd.to_datetime("2022-12-31"))]
trainX = train.drop('^DJI', axis=1)
trainY = pd.DataFrame({'^DJI': train['^DJI']})
valX = val.drop('^DJI', axis=1)
valY = pd.DataFrame({'^DJI': val['^DJI']})
testX = test.drop('^DJI', axis=1)
testY = pd.DataFrame({'^DJI': test['^DJI']})


# performance evaluation function
def evaluate(ds, optimization):
    dowjones_hpr = (Returns['^DJI'][-1] - Returns['^DJI'][0]) / Returns['^DJI'][0]
    portfolio_hpr = (ds[optimization][-1] - ds[optimization][0]) / ds[optimization][0]
    portfolio_active_return = portfolio_hpr - dowjones_hpr
    portfolio_tracking_error = np.std(ds[optimization] - Returns['^DJI'])
    info_ratio = portfolio_active_return / portfolio_tracking_error

    chart = pd.DataFrame()
    chart['^DJI'] = ds['^DJI']
    chart[f'{optimization}'] = ds[f'{optimization}']
    chart.plot()
    plt.title(f"{optimization} optimization")
    plt.show()

    print("Portfolio Evaluation")
    print("*" * 40)
    print("Active Return:", round(portfolio_active_return, 5))
    print("Tracking Error:", round(portfolio_tracking_error * 10000), "bps")
    print("Information Ratio:", round(info_ratio, 5))
    print("Returns RMSE:", mean_squared_error(ds[optimization], ds['^DJI'], squared=False))
    print("\n")
    pass


# beta regression function
def beta_reg(stock):
    const = np.polyfit(Returns['^DJI'], Returns[f'{stock}'], 1)[1]
    Beta = np.polyfit(Returns['^DJI'], Returns[f'{stock}'], 1)[0]
    model = sm.OLS(Returns[f'{stock}'], sm.add_constant(Returns['^DJI'])).fit()

    print(model.summary())
    print(f'\n{stock} Beta:', '{:.4f}'.format(Beta))

    plt.scatter(Returns['^DJI'], Returns[f'{stock}'], color='pink')
    plt.plot(Returns['^DJI'], Beta * Returns['^DJI'] + const, color='purple')
    plt.xlabel('^DJI')
    plt.ylabel(f'{stock}')
    plt.show()
    pass
