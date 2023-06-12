import staging as stage
import pandas as pd
import numpy as np
from scipy.optimize import nnls

# portfolio allocation with Non-Negative Least Squares (NNLS)
residual = nnls(stage.trainX, stage.trainY['^DJI'])
leverage = sum(residual[0])
weights = dict(zip(stage.trainX.columns, residual[0] / leverage))
allocation = pd.DataFrame({'Component': list(weights.keys()),
                           'Weight(%)': np.multiply(list(weights.values()), 100)}).sort_values('Weight(%)', ascending=False)
allocation.set_index('Component', inplace=True)
allocation.reset_index(inplace=True)
print("*" * 10, "Non-Negative Least Squares (NNLS) Optimization", "*" * 10)
print("\nPortfolio Allocation:")
print(allocation)
print("\nLeverage Factor:", leverage, "\n")
pass

# portfolio optimization evaluation
stage.valY['NNLS'] = leverage * stage.valX.dot(list(weights.values()))
stage.evaluate(stage.valY, 'NNLS')
stage.testY['NNLS'] = leverage * stage.testX.dot(list(weights.values()))
stage.evaluate(stage.testY, 'NNLS')