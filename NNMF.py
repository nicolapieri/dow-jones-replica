import staging as stage
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF

# portfolio allocation with Non-Negative Matrix Factorization (NNMF)
TRAIN = stage.Closes[(stage.Closes.index >= stage.pd.to_datetime("1993-01-01")) & (stage.Closes.index <= pd.to_datetime("2021-12-31"))]
TRAINX = TRAIN.drop('^DJI', axis=1)
coeffs = NMF(n_components=1).fit(TRAINX).components_.tolist()[0]
leverage = sum(coeffs)
weights = dict(zip(list(TRAINX.columns), np.divide(coeffs, sum(coeffs))))
allocation = pd.DataFrame({'Component': list(weights.keys()),
                           'Weight(%)': np.multiply(list(weights.values()), 100)}).sort_values('Weight(%)', ascending=False)
allocation.set_index('Component', inplace=True)
allocation.reset_index(inplace=True)
print("*" * 10, "Non-Negative Matrix Factorization (NNMF) Optimization", "*" * 10)
print("\nPortfolio Allocation:")
print(allocation)
print('\n')
pass

# portfolio optimization evaluation
stage.valY['NNMF'] = stage.valX.dot(list(weights.values()))
stage.evaluate(stage.valY, 'NNMF')
stage.testY['NNMF'] = stage.testX.dot(list(weights.values()))
stage.evaluate(stage.testY, 'NNMF')