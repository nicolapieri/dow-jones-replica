import pandas as pd
import yfinance as yf

# setting time interval
start_date, end_date = "1992-01-01", "2022-12-31"

# downloading dow jones index and stocks datasets
DJI = pd.DataFrame(yf.download("^DJI", start=start_date, end=end_date))
MMM = pd.DataFrame(yf.download("MMM", start=start_date, end=end_date))
AXP = pd.DataFrame(yf.download("AXP", start=start_date, end=end_date))
AMGN = pd.DataFrame(yf.download("AMGN", start=start_date, end=end_date))
AAPL = pd.DataFrame(yf.download("AAPL", start=start_date, end=end_date))
BA = pd.DataFrame(yf.download("BA", start=start_date, end=end_date))
CAT = pd.DataFrame(yf.download("CAT", start=start_date, end=end_date))
CVX = pd.DataFrame(yf.download("CVX", start=start_date, end=end_date))
CSCO = pd.DataFrame(yf.download("CSCO", start=start_date, end=end_date))
KO = pd.DataFrame(yf.download("KO", start=start_date, end=end_date))
DOW = pd.DataFrame(yf.download("DOW", start=start_date, end=end_date))
GS = pd.DataFrame(yf.download("GS", start=start_date, end=end_date))
HD = pd.DataFrame(yf.download("HD", start=start_date, end=end_date))
HON = pd.DataFrame(yf.download("HON", start=start_date, end=end_date))
INTC = pd.DataFrame(yf.download("INTC", start=start_date, end=end_date))
IBM = pd.DataFrame(yf.download("IBM", start=start_date, end=end_date))
JNJ = pd.DataFrame(yf.download("JNJ", start=start_date, end=end_date))
JPM = pd.DataFrame(yf.download("JPM", start=start_date, end=end_date))
MCD = pd.DataFrame(yf.download("MCD", start=start_date, end=end_date))
MRK = pd.DataFrame(yf.download("MRK", start=start_date, end=end_date))
MSFT = pd.DataFrame(yf.download("MSFT", start=start_date, end=end_date))
NKE = pd.DataFrame(yf.download("NKE", start=start_date, end=end_date))
PG = pd.DataFrame(yf.download("PG", start=start_date, end=end_date))
CRM = pd.DataFrame(yf.download("CRM", start=start_date, end=end_date))
TRV = pd.DataFrame(yf.download("TRV", start=start_date, end=end_date))
UNH = pd.DataFrame(yf.download("UNH", start=start_date, end=end_date))
VZ = pd.DataFrame(yf.download("VZ", start=start_date, end=end_date))
V = pd.DataFrame(yf.download("V", start=start_date, end=end_date))
WBA = pd.DataFrame(yf.download("WBA", start=start_date, end=end_date))
WMT = pd.DataFrame(yf.download("MSFT", start=start_date, end=end_date))
DIS = pd.DataFrame(yf.download("MSFT", start=start_date, end=end_date))

# computing daily returns over adjusted closing prices
DJI['DJI Adj Returns'] = (DJI['Adj Close'] - DJI['Adj Close'].shift(1)) / DJI['Adj Close'].shift(1)
DJI['MMM Adj Returns'] = (MMM['Adj Close'] - MMM['Adj Close'].shift(1)) / MMM['Adj Close'].shift(1)
DJI['AXP Adj Returns'] = (AXP['Adj Close'] - AXP['Adj Close'].shift(1)) / AXP['Adj Close'].shift(1)
DJI['AMGN Adj Returns'] = (AMGN['Adj Close'] - AMGN['Adj Close'].shift(1)) / AMGN['Adj Close'].shift(1)
DJI['MMM Adj Returns'] = (MMM['Adj Close'] - MMM['Adj Close'].shift(1)) / MMM['Adj Close'].shift(1)
DJI['MMM Adj Returns'] = (MMM['Adj Close'] - MMM['Adj Close'].shift(1)) / MMM['Adj Close'].shift(1)
DJI['MMM Adj Returns'] = (MMM['Adj Close'] - MMM['Adj Close'].shift(1)) / MMM['Adj Close'].shift(1)
DJI['MMM Adj Returns'] = (MMM['Adj Close'] - MMM['Adj Close'].shift(1)) / MMM['Adj Close'].shift(1)
DJI['MMM Adj Returns'] = (MMM['Adj Close'] - MMM['Adj Close'].shift(1)) / MMM['Adj Close'].shift(1)
DJI['MMM Adj Returns'] = (MMM['Adj Close'] - MMM['Adj Close'].shift(1)) / MMM['Adj Close'].shift(1)
DJI['MMM Adj Returns'] = (MMM['Adj Close'] - MMM['Adj Close'].shift(1)) / MMM['Adj Close'].shift(1)

print(DJI.info())
