import staging as stage
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF

# portfolio allocation with Non-Negative Matrix Factorization (NNMF)
NNMF_coeffs = NMF(n_components=1).fit(stage.X).components_.tolist()[0]
NNMF_weights = dict(zip(list(stage.X.columns), np.divide(NNMF_coeffs, sum(NNMF_coeffs))))
NNMF_allocation = pd.DataFrame({'Component': list(NNMF_weights.keys()),
                                'NNMF-30wg(%)': np.multiply(list(NNMF_weights.values()), 100)}).sort_values(
    'NNMF-30wg(%)', ascending=False)
NNMF_allocation.set_index('Component', inplace=True)
NNMF_allocation.reset_index(inplace=True)

# NNMF optimization evaluation
stage.Opt['NNMF'] = stage.X.dot(list(NNMF_weights.values()))
stage.evaluate(stage.Opt, 'NNMF')
