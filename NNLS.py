import staging as stage
import pandas as pd
import numpy as np
from scipy.optimize import nnls

# training with Non-Negative Least Squares (NNLS)
NNLS_residual = nnls(stage.X, stage.Y['^DJI'])
NNLS_leverage = sum(NNLS_residual[0])
NNLS_weights = dict(zip(stage.X.columns, NNLS_residual[0] / NNLS_leverage))
NNLS_allocation = pd.DataFrame({'Component': list(NNLS_weights.keys()),
                                'NNLS-30wg(%)': np.multiply(list(NNLS_weights.values()), 100)}).sort_values(
    'NNLS-30wg(%)', ascending=False)
NNLS_allocation.set_index('Component', inplace=True)
NNLS_allocation.reset_index(inplace=True)

# testing NNLS portfolio optimization
stage.Opt['NNLS'] = NNLS_leverage * stage.X.dot(list(NNLS_weights.values()))
stage.evaluate(stage.Opt, 'NNLS')
