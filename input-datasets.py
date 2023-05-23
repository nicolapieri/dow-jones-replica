import yfinance as yf
import pandas as pd

# setting time interval
start_date, end_date = "1992-01-01", "2022-12-31"

# downloading dow jones index dataset
pd.DataFrame(yf.download("^DJI", start=start_date, end=end_date))\
    .to_csv(".\\financial-markets-data\\^DJI.csv", encoding='utf-8')

# declaring dow jones composition (https://en.wikipedia.org/wiki/Historical_components_of_the_Dow_Jones_Industrial_Average#May_6,_1991)
comp2020 = ["MMM", "AXP", "AMGN", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO", "DOW",
            "GS", "HD", "HON", "INTC", "IBM", "JNJ", "JPM", "MCD", "MRK", "MSFT",
            "NKE", "PG", "CRM", "TRV", "UNH", "VZ", "V", "WBA", "WMT", "DIS"]

# downloading dow jones stocks datasets
for stock in comp2020:
    pd.DataFrame(yf.download(stock, start=start_date, end=end_date)) \
        .to_csv(f".\\financial-markets-data\\{stock}.csv", encoding='utf-8')
