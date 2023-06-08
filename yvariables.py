import pandas as pd
import yfinance as yf

# https://www.spglobal.com/spdji/en/documents/methodologies/methodology-dj-averages.pdf

# setting time interval
start_date, end_date = "1993-01-01", "2022-12-31"

# getting Dow Jones index timeseries
DJI = pd.DataFrame(yf.download("^DJI", start=start_date, end=end_date))
DJI.to_csv(".\\ydatasets\\^DJI.csv", encoding='utf-8')

# declaring Dow Jones compositions (https://en.wikipedia.org/wiki/Historical_components_of_the_Dow_Jones_Industrial_Average)
DJIA_components = ["MMM", "AXP", "AMGN", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO",
                   "HD", "HON", "INTC", "IBM", "JNJ", "JPM", "MCD", "MRK", "MSFT",
                   "NKE", "PG", "TRV", "UNH", "VZ", "WBA", "WMT", "DIS"]  # excluded CRM, DOW, GS and V for missing values

# declaring components and weights (https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average)
DJIA_weights = {'AAPL': 0.0284, 'AMGN': 0.0548,
                'AXP': 0.0302, 'BA': 0.0336,
                'CAT': 0.0452, 'CSCO': 0.0096,
                'CVX': 0.0350, 'DIS': 0.0189,
                'HD': 0.0627, 'HON': 0.0417,
                'IBM': 0.0286, 'INTC': 0.0057,
                'JNJ': 0.0343, 'JPM': 0.0261,
                'KO': 0.0122, 'MCD': 0.0524,
                'MMM': 0.0241, 'MRK': 0.0210,
                'MSFT': 0.0488, 'NKE': 0.0213,
                'PG': 0.0286, 'TRV': 0.0362,
                'UNH': 0.1029, 'VZ': 0.0073,
                'WBA': 0.0079, 'WMT': 0.0294}

# creating closes and returns dataframes
Closes = pd.DataFrame()
Returns = pd.DataFrame()
for symbol in DJIA_components:
    globals()[symbol] = pd.DataFrame(yf.download(symbol, start=start_date, end=end_date))
    Closes[symbol] = globals()[symbol].loc[:, 'Adj Close']
    Returns[symbol] = (globals()[symbol].loc[:, 'Adj Close'] - globals()[symbol].loc[:, 'Adj Close'].shift(1)) / globals()[symbol].loc[:, 'Adj Close'].shift(1)
Closes['avg26'] = Closes.sum(axis=1)
Closes['^DJI'] = DJI['Adj Close']
Returns.drop(index=Returns.index[0], axis=0, inplace=True)
Returns['avg26'] = Returns.mean(axis=1)
Returns['^DJI'] = (DJI['Adj Close'] - DJI['Adj Close'].shift(1)) / DJI['Adj Close'].shift(1)