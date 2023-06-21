import staging as stage
import pandas as pd
import numpy as np
from scipy.optimize import nnls

# portfolio allocation with Non-Negative Least Squares (NNLS)
residual_NNLS = nnls(stage.trainX, stage.trainY['^DJI'])
leverage_NNLS = sum(residual_NNLS[0])
weights_NNLS = dict(zip(stage.trainX.columns, residual_NNLS[0] / leverage_NNLS))
allocation_NNLS = pd.DataFrame({'Component': list(weights_NNLS.keys()),
                                'NNLSweight(%)': np.multiply(list(weights_NNLS.values()), 100)}).sort_values('NNLSweight(%)', ascending=False)
allocation_NNLS.set_index('Component', inplace=True)
allocation_NNLS.reset_index(inplace=True)

# NNLS optimization evaluation
stage.testY['NNLS'] = leverage_NNLS * stage.testX.dot(list(weights_NNLS.values()))
stage.evaluate(stage.testY, 'NNLS')

print(allocation_NNLS)
print("-" * 50)
print("Leverage Factor:", leverage_NNLS)
