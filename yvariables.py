import pandas as pd
import yfinance as yf

# setting time interval
start_date, end_date = "1992-01-01", "2022-12-31"

# getting Dow Jones index timeseries
DJI = pd.DataFrame(yf.download("^DJI", start=start_date, end=end_date))
DJI.to_csv(".\\ydatasets\\^DJI.csv", encoding='utf-8')

# declaring Dow Jones compositions (https://en.wikipedia.org/wiki/Historical_components_of_the_Dow_Jones_Industrial_Average)
composition = ["MMM", "AXP", "AMGN", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO", "DOW",
               "GS", "HD", "HON", "INTC", "IBM", "JNJ", "JPM", "MCD", "MRK", "MSFT",
               "NKE", "PG", "CRM", "TRV", "UNH", "VZ", "V", "WBA", "WMT", "DIS"]

# creating returns dataframe
Returns = pd.DataFrame()
Returns['DJI_Returns'] = (DJI['Adj Close'] - DJI['Adj Close'].shift(1)) / DJI['Adj Close'].shift(1)
for symbol in composition:
    globals()[symbol] = pd.DataFrame(yf.download(symbol, start=start_date, end=end_date))
    Returns[symbol + '_Returns'] = (globals()[symbol].loc[:, 'Adj Close'] - globals()[symbol].loc[:, 'Adj Close'].shift(1)) / globals()[symbol].loc[:, 'Adj Close'].shift(1)
Returns.drop(index=Returns.index[0], axis=0, inplace=True)
