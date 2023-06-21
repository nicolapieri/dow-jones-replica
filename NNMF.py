import staging as stage
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF

# portfolio allocation with Non-Negative Matrix Factorization (NNMF)
coeffs = NMF(n_components=1).fit(stage.trainX).components_.tolist()[0]
leverage_NNMF = sum(coeffs)
weights_NNMF = dict(zip(list(stage.trainX.columns), np.divide(coeffs, sum(coeffs))))
allocation_NNMF = pd.DataFrame({'Component': list(weights_NNMF.keys()),
                                'NNMFweight(%)': np.multiply(list(weights_NNMF.values()),100)}).sort_values('NNMFweight(%)', ascending=False)
allocation_NNMF.set_index('Component', inplace=True)
allocation_NNMF.reset_index(inplace=True)

# NNMF optimization evaluation
stage.testY['NNMF'] = stage.testX.dot(list(weights_NNMF.values()))
stage.evaluate(stage.testY, 'NNMF')

print(allocation_NNMF)
print("-" * 50)
print(f"Leverage Factor: {leverage_NNMF}")
