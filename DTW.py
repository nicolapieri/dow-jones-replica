import staging as stage
import pandas as pd
import numpy as np
from tslearn.metrics import dtw

# portfolio allocation with Dynamic Time Warping (DTW)
distances = stage.trainX.apply(lambda x: dtw(x, stage.trainY))
weights = (1 / distances / sum(1 / distances)).to_dict()
allocation_DTW = pd.DataFrame({'Component': list(weights.keys()),
                               'DTWweight(%)': np.multiply(list(weights.values()), 100)}).sort_values('DTWweight(%)', ascending=False)
allocation_DTW.set_index('Component', inplace=True)
allocation_DTW.reset_index(inplace=True)

# DTW optimization evaluation
stage.testY['DTW'] = stage.testX.dot(list(weights.values()))
stage.evaluate(stage.testY, 'DTW')
