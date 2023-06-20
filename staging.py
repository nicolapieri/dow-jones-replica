import yfinance as yf
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# setting time interval
start_date, end_date = "2020-09-01", "2023-05-31"

# getting Dow Jones index timeseries
DJI = pd.DataFrame(yf.download("^DJI", start=start_date, end=end_date))

# declaring Dow Jones compositions (https://en.wikipedia.org/wiki/Historical_components_of_the_Dow_Jones_Industrial_Average)
DJI_components = ["MMM", "AXP", "AMGN", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO",
                  "HD", "HON", "INTC", "IBM", "JNJ", "JPM", "MCD", "MRK", "MSFT",
                  "NKE", "PG", "TRV", "UNH", "VZ", "WBA", "WMT", "DIS", "CRM", "DOW", "GS", "V"]

# creating closes and returns dataframes
Closes_std = pd.DataFrame()
for symbol in tqdm(DJI_components):
    ticker = pd.DataFrame(yf.download(symbol, start=start_date, end=end_date))
    Closes_std[symbol] = ticker.loc[:, 'Adj Close']
cols = list(Closes_std)
for column in cols:
    fitted_t = StandardScaler().fit(np.array(Closes_std[column]).reshape(len(Closes_std[column]), 1))
    Closes_std[column] = fitted_t.transform(np.array(Closes_std[column]).reshape(len(Closes_std[column]), 1))
fitted_C = StandardScaler().fit(np.array(DJI['Adj Close']).reshape(len(DJI['Adj Close']), 1))
Closes_std['^DJI'] = fitted_C.transform(np.array(DJI['Adj Close']).reshape(len(DJI['Adj Close']), 1))

# train, val, test sets split
train = Closes_std[
    (Closes_std.index >= pd.to_datetime("2020-09-01")) & (Closes_std.index <= pd.to_datetime("2022-12-31"))]
test = Closes_std[
    (Closes_std.index >= pd.to_datetime("2023-01-01")) & (Closes_std.index <= pd.to_datetime("2023-05-31"))]
trainX = train.drop('^DJI', axis=1)
trainY = pd.DataFrame({'^DJI': train['^DJI']})
testX = test.drop('^DJI', axis=1)
testY = pd.DataFrame({'^DJI': test['^DJI']})
print('TOT:', len(Closes_std))
print('Train:', len(train))
print('Test:', len(test))

# performance evaluation function
opt_performances = {}


def evaluate(ds, optimization):
    dowjones_hpr = (Closes_std['^DJI'][-1] - Closes_std['^DJI'][0]) / Closes_std['^DJI'][0]
    portfolio_hpr = (ds[optimization][-1] - ds[optimization][0]) / ds[optimization][0]
    portfolio_active_return = portfolio_hpr - dowjones_hpr
    portfolio_tracking_error = np.std(ds[optimization] - Closes_std['^DJI'])
    info_ratio = portfolio_active_return / portfolio_tracking_error
    opt_performances[optimization] = round(portfolio_tracking_error * 10000)

    chart = pd.DataFrame()
    chart['^DJI'] = ds['^DJI']
    chart[f'{optimization}'] = ds[f'{optimization}']
    chart.plot()
    plt.title(f"{optimization} optimization")
    plt.show()

    print(f"\n{optimization} Portfolio Evaluation")
    print("*" * 40)
    print("Active Return:", round(portfolio_active_return, 5), "%")
    print("Tracking Error:", round(portfolio_tracking_error * 10000), "bps")
    print("Information Ratio:", round(info_ratio, 5))
    print("Returns RMSE:", mean_squared_error(ds[optimization], ds['^DJI'], squared=False))
    pass
