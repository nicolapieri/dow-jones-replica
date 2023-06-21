import yfinance as yf
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# declaring Dow Jones composition (https://en.wikipedia.org/wiki/Historical_components_of_the_Dow_Jones_Industrial_Average)
components = ["MMM", "AXP", "AMGN", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO",
              "HD", "HON", "INTC", "IBM", "JNJ", "JPM", "MCD", "MRK", "MSFT",
              "NKE", "PG", "TRV", "UNH", "VZ", "WBA", "WMT", "DIS", "CRM", "DOW", "GS", "V"]

# creating adjusted closes dataframe
Closes = pd.DataFrame()
for ticker in tqdm(components):
    stock = pd.DataFrame(yf.download(ticker, start="2020-09-01", end="2023-05-31"))
    Closes[ticker] = stock.loc[:, 'Adj Close']

Closes['^DJI'] = pd.DataFrame(yf.download("^DJI", start="2020-09-01", end="2023-05-31"))['Adj Close']

# train-test sets split
train_start, train_end = "2020-09-01", "2022-12-31"
test_start, test_end = "2023-01-01", "2023-05-31"

trainX = Closes[(Closes.index >= pd.to_datetime(train_start)) & (Closes.index <= pd.to_datetime(train_end))].drop(
    '^DJI', axis=1)
trainY = pd.DataFrame({'^DJI': Closes[
    (Closes.index >= pd.to_datetime(train_start)) & (Closes.index <= pd.to_datetime(train_end))]['^DJI']})
testX = Closes[(Closes.index >= pd.to_datetime(test_start)) & (Closes.index <= pd.to_datetime(test_end))].drop('^DJI',
                                                                                                               axis=1)
testY = pd.DataFrame(
    {'^DJI': Closes[(Closes.index >= pd.to_datetime(test_start)) & (Closes.index <= pd.to_datetime(test_end))]['^DJI']})

print('\n')
print(f'Considering the time period from {train_start} to {test_end},', f'{len(Closes)} observations')
print(f'Training period from {train_start} to {train_end},', f'{len(trainX)} observations')
print(f'Testing period from {test_start} to {test_end},', f'{len(testX)} observations')

# performance evaluation function
opt_performances = {}


def evaluate(ds, optimization):
    portfolio_hpr = (ds[optimization][-1] - ds[optimization][0]) / ds[optimization][0]
    dowjones_hpr = (ds['^DJI'][-1] - ds['^DJI'][0]) / ds['^DJI'][0]
    portfolio_tracking_error = np.std(ds[optimization].pct_change().dropna() - ds['^DJI'].pct_change().dropna())
    rooted_mse = mean_squared_error(ds[optimization].pct_change().dropna(), ds['^DJI'].pct_change().dropna(),
                                    squared=False)
    portfolio_active_return = (portfolio_hpr - dowjones_hpr)
    info_ratio = portfolio_active_return / portfolio_tracking_error
    opt_performances[optimization] = round(portfolio_tracking_error * 10000, 4)

    chart = pd.DataFrame()
    chart['Daily Deviation (bps)'] = round((ds[optimization].pct_change().dropna() - ds['^DJI'].pct_change().dropna()) * 10000, 4)
    chart.plot(color='purple')
    plt.title(f"{optimization} optimization")
    plt.show()

    print(f"\nPortfolio Evaluation - {optimization}")
    print("-" * 50)
    print(f"Average Tracking Error: {round(portfolio_tracking_error * 10000, 4)} bps")
    print(f"Information Ratio: {round(info_ratio, 4)}")
    print(f"Rooted MSE from ^DJI: {round(rooted_mse * 100, 4)} %")
    print(f"Portfolio Active Return (testing period): {round(portfolio_active_return * 100, 4)} %")
    print("-" * 50)
    pass
