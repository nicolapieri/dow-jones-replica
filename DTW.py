import staging as stage
import pandas as pd
import numpy as np
from tslearn.metrics import dtw

# portfolio allocation with Dynamic Time Warping (DTW)
DTWtrain_distances = stage.trainX.apply(lambda x: dtw(x, stage.trainY))
DTWtrain_weights = (1 / DTWtrain_distances / sum(1 / DTWtrain_distances)).to_dict()
DTWtrain_allocation = pd.DataFrame({'Component': list(DTWtrain_weights.keys()),
                                    'DTW-30wg(%)': np.multiply(list(DTWtrain_weights.values()), 100)}).sort_values('DTW-30wg(%)', ascending=False)
DTWtrain_allocation.set_index('Component', inplace=True)
DTWtrain_allocation.reset_index(inplace=True)

# validation with top 10 weighted stocks
DTWval_distances = stage.valX[DTWtrain_allocation['Component'][0:10]].apply(lambda x: dtw(x, stage.valY))
DTWval_weights = (1 / DTWval_distances / sum(1 / DTWval_distances)).to_dict()
DTWval_allocation = pd.DataFrame({'Component': list(DTWval_weights.keys()),
                                  'DTW-10wg(%)': np.multiply(list(DTWval_weights.values()), 100)}).sort_values('DTW-10wg(%)', ascending=False)
DTWval_allocation.set_index('Component', inplace=True)
DTWval_allocation.reset_index(inplace=True)

# DTW portfolio optimization evaluation
stage.testY['DTW'] = stage.testX[DTWval_allocation['Component'][0:10]].dot(list(DTWval_weights.values()))
stage.evaluate(stage.testY, 'DTW')

print(DTWval_allocation)
print("-" * 50)
print("Leverage Factor: none")
