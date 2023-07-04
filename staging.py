import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# declaring Dow Jones composition
components = ["MMM", "AXP", "AMGN", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO",
              "HD", "HON", "INTC", "IBM", "JNJ", "JPM", "MCD", "MRK", "MSFT",
              "NKE", "PG", "TRV", "UNH", "VZ", "WBA", "WMT", "DIS", "CRM", "DOW", "GS", "V"]

# creating adjusted closes dataframe
X = pd.DataFrame()
for ticker in components:
    globals()[ticker] = pd.DataFrame(yf.download(ticker, start="2020-09-01", end="2023-05-31"))
    X[ticker] = globals()[ticker].loc[:, 'Adj Close']

Y = pd.DataFrame()
Y['^DJI'] = pd.DataFrame(yf.download("^DJI", start="2020-09-01", end="2023-05-31"))['Adj Close']
Opt = pd.DataFrame()
Opt['^DJI'] = Y['^DJI']

# performance evaluation function
opt_terrors = {}


def evaluate(ds, optimization):
    portfolio_hpr = (ds[optimization][-1] - ds[optimization][0]) / ds[optimization][0]
    dowjones_hpr = (Y['^DJI'][-1] - Y['^DJI'][0]) / Y['^DJI'][0]
    portfolio_tracking_error = np.std(ds[optimization].pct_change().dropna() - Y['^DJI'].pct_change().dropna())
    portfolio_active_return = (portfolio_hpr - dowjones_hpr)
    info_ratio = portfolio_active_return / portfolio_tracking_error
    opt_terrors[optimization] = round(portfolio_tracking_error * 10000, 4)

    plt.style.use('Solarize_Light2')
    chart = pd.DataFrame()
    chart['Portfolio Tracking Error'] = round(
        (ds[optimization].pct_change().dropna() - Y['^DJI'].pct_change().dropna()) * 10000, 4)
    chart[(chart.index >= pd.to_datetime('2023-05-01')) & (chart.index <= pd.to_datetime('2023-05-31'))].plot(
        color='purple')
    plt.title(f"{optimization} optimization")
    plt.hlines(y=5, xmin='2023-05-01', xmax="2023-05-31", colors='indigo', linestyle='dashed')
    plt.hlines(y=-5, xmin='2023-05-01', xmax="2023-05-31", colors='indigo', linestyle='dashed')
    plt.ylabel('Basis Points (bps)')
    plt.show()

    print(f"\nPortfolio Evaluation - {optimization}")
    print("-" * 50)
    print(f"Standard Tracking Error: {round(portfolio_tracking_error * 10000, 4)} bps")
    print(f"Information Ratio: {round(info_ratio, 4)}")
    print(f"Portfolio Active Return: {round(portfolio_active_return * 100, 4)} %")
    print("-" * 50)
    pass
