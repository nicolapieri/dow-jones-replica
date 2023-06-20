import staging as stage
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
from sklearn.decomposition import NMF

# recreating datasets because NNMF only accept positive values
Closes = pd.DataFrame()
for symbol in tqdm(stage.DJI_components):
    yahoodata = pd.DataFrame(yf.download(symbol, start=stage.start_date, end=stage.end_date))
    Closes[symbol] = yahoodata.loc[:, 'Adj Close']
Closes.drop(index=Closes.index[0], axis=0, inplace=True)
Closes['^DJI'] = stage.DJI['Adj Close']
TRAIN = Closes[(Closes.index >= pd.to_datetime("1993-01-01")) & (Closes.index <= pd.to_datetime("2021-12-31"))]
TRAINX = TRAIN.drop('^DJI', axis=1)

# portfolio allocation with Non-Negative Matrix Factorization (NNMF)
coeffs = NMF(n_components=1).fit(TRAINX).components_.tolist()[0]
leverage = sum(coeffs)
weights = dict(zip(list(TRAINX.columns), np.divide(coeffs, sum(coeffs))))
allocation_NNMF = pd.DataFrame({'Component': list(weights.keys()),
                                'NNMFweight(%)': np.multiply(list(weights.values()),100)}).sort_values('NNMFweight(%)', ascending=False)
allocation_NNMF.set_index('Component', inplace=True)
allocation_NNMF.reset_index(inplace=True)

# NNMF optimization evaluation
stage.testY['NNMF'] = stage.testX.dot(list(weights.values()))
stage.evaluate(stage.testY, 'NNMF')
