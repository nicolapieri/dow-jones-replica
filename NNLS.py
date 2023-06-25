import staging as stage
import pandas as pd
import numpy as np
from scipy.optimize import nnls

# training with Non-Negative Least Squares (NNLS)
NNLStrain_residual = nnls(stage.trainX, stage.trainY['^DJI'])
NNLStrain_leverage = sum(NNLStrain_residual[0])
NNLStrain_weights = dict(zip(stage.trainX.columns, NNLStrain_residual[0] / NNLStrain_leverage))
NNLStrain_allocation = pd.DataFrame({'Component': list(NNLStrain_weights.keys()),
                                     'NNLS-30wg(%)': np.multiply(list(NNLStrain_weights.values()), 100)}).sort_values('NNLS-30wg(%)', ascending=False)
NNLStrain_allocation.set_index('Component', inplace=True)
NNLStrain_allocation.reset_index(inplace=True)

# validation with top 10 weighted stocks
NNLSval_residual = nnls(stage.valX[NNLStrain_allocation['Component'][0:10]], stage.valY['^DJI'])
NNLSval_leverage = sum(NNLSval_residual[0])
NNLSval_weights = dict(zip(stage.valX[NNLStrain_allocation['Component'][0:10]].columns, NNLSval_residual[0] / NNLSval_leverage))
NNLSval_allocation = pd.DataFrame({'Component': list(NNLSval_weights.keys()),
                                   'NNLS-10wg(%)': np.multiply(list(NNLSval_weights.values()), 100)}).sort_values('NNLS-10wg(%)', ascending=False)
NNLSval_allocation.set_index('Component', inplace=True)
NNLSval_allocation.reset_index(inplace=True)

# testing NNLS portfolio optimization
stage.testY['NNLS'] = NNLSval_leverage * stage.testX[NNLSval_allocation['Component'][0:10]].dot(list(NNLSval_weights.values()))
stage.evaluate(stage.testY, 'NNLS')

print(NNLSval_allocation)
print("-" * 50)
print("Leverage Factor:", NNLSval_leverage)
