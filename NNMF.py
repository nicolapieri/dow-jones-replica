import staging as stage
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF

# portfolio allocation with Non-Negative Matrix Factorization (NNMF)
NNMFtrain_coeffs = NMF(n_components=1).fit(stage.trainX).components_.tolist()[0]
NNMFtrain_weights = dict(zip(list(stage.trainX.columns), np.divide(NNMFtrain_coeffs, sum(NNMFtrain_coeffs))))
NNMFtrain_allocation = pd.DataFrame({'Component': list(NNMFtrain_weights.keys()),
                                     'NNMF-30wg(%)': np.multiply(list(NNMFtrain_weights.values()),100)}).sort_values('NNMF-30wg(%)', ascending=False)
NNMFtrain_allocation.set_index('Component', inplace=True)
NNMFtrain_allocation.reset_index(inplace=True)

# validation with top 10 weighted stocks
NNMFval_coeffs = NMF(n_components=1).fit(stage.valX[NNMFtrain_allocation['Component'][0:10]]).components_.tolist()[0]
NNMFval_weights = dict(zip(list(stage.valX[NNMFtrain_allocation['Component'][0:10]].columns), np.divide(NNMFval_coeffs, sum(NNMFval_coeffs))))
NNMFval_allocation = pd.DataFrame({'Component': list(NNMFval_weights.keys()),
                                   'NNMF-10wg(%)': np.multiply(list(NNMFval_weights.values()),100)}).sort_values('NNMF-10wg(%)', ascending=False)
NNMFval_allocation.set_index('Component', inplace=True)
NNMFval_allocation.reset_index(inplace=True)

# NNMF optimization evaluation
stage.testY['NNMF'] = stage.testX[NNMFval_allocation['Component'][0:10]].dot(list(NNMFval_weights.values()))
stage.evaluate(stage.testY, 'NNMF')

print(NNMFval_allocation)
print("-" * 50)
print(f"Leverage Factor: none")
