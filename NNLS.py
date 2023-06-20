import staging as stage
import pandas as pd
import numpy as np
from scipy.optimize import nnls

# portfolio allocation with Non-Negative Least Squares (NNLS)
residual = nnls(stage.trainX, stage.trainY['^DJI'])
leverage = sum(residual[0])
weights = dict(zip(stage.trainX.columns, residual[0] / leverage))
allocation_NNLS = pd.DataFrame({'Component': list(weights.keys()),
                                'NNLSweight(%)': np.multiply(list(weights.values()), 100)}).sort_values('NNLSweight(%)', ascending=False)
allocation_NNLS.set_index('Component', inplace=True)
allocation_NNLS.reset_index(inplace=True)

# NNLS optimization evaluation
stage.testY['NNLS'] = leverage * stage.testX.dot(list(weights.values()))
stage.evaluate(stage.testY, 'NNLS')
print("\nLeverage Factor:", leverage)
