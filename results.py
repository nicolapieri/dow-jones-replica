import staging as stage
import NNLS
import PCRR
import DTW
import NNMF
import PSO
from tqdm import tqdm
import pandas as pd
import numpy as np
import yfinance as yf

# creating dataframe for stocks betas
Closes_ret = pd.DataFrame()
for symbol in tqdm(stage.DJI_components):
    yahoodata = pd.DataFrame(yf.download(symbol, start=stage.start_date, end=stage.end_date))
    Closes_ret[symbol] = (yahoodata.loc[:, 'Adj Close'] - yahoodata.loc[:, 'Adj Close'].shift(1)) / yahoodata.loc[:, 'Adj Close'].shift(1)
Closes_ret.drop(index=Closes_ret.index[0], axis=0, inplace=True)
Closes_ret['^DJI'] = (stage.DJI['Adj Close'] - stage.DJI['Adj Close'].shift(1)) / stage.DJI['Adj Close'].shift(1)

B = {}
for stock in stage.DJI_components:
    B[stock] = np.polyfit(Closes_ret['^DJI'], Closes_ret[f'{stock}'], 1)[0]
betas = pd.DataFrame(B.items(), columns=['Component', 'Beta'])

# joining all the different allocations
# all_comparison['AVGweight(%)'] = all_comparison.mean(axis=1)
all_comparison = pd.merge(pd.merge(pd.merge(pd.merge(NNLS.allocation_NNLS,
                                                     PCRR.allocation_PCRR, on='Component'),
                                            DTW.allocation_DTW, on='Component'),
                                   NNMF.allocation_NNMF, on='Component'),
                          PSO.allocation_PSO, on='Component')

# creating final results dataframe
results = pd.merge(all_comparison, betas, on='Component')

print(results)
print('\nTracking Errors:', stage.opt_performances)
