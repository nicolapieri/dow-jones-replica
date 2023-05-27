import pandas as pd
import yfinance as yf

# setting time interval
start_date, end_date = "1992-01-01", "2022-12-31"

# getting Dow Jones index timeseries
DJI = pd.DataFrame(yf.download("^DJI", start=start_date, end=end_date))
DJI.to_csv(".\\ydatasets\\^DJI.csv", encoding='utf-8')

# declaring Dow Jones composition (https://en.wikipedia.org/wiki/Historical_components_of_the_Dow_Jones_Industrial_Average)
comp_20_08 = ["MMM", "AXP", "AMGN", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO", "DOW",
              "GS", "HD", "HON", "INTC", "IBM", "JNJ", "JPM", "MCD", "MRK", "MSFT",
              "NKE", "PG", "CRM", "TRV", "UNH", "VZ", "V", "WBA", "WMT", "DIS"]

# getting Dow Jones stocks timeseries
for symbol in comp_20_08:
    globals()[symbol] = pd.DataFrame(yf.download(symbol, start=start_date, end=end_date))
    globals()[symbol].to_csv(f".\\ydatasets\\{symbol}.csv", encoding='utf-8')

