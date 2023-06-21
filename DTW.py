import staging as stage
import pandas as pd
import numpy as np
from tslearn.metrics import dtw

# portfolio allocation with Dynamic Time Warping (DTW)
distances_DTW = stage.trainX.apply(lambda x: dtw(x, stage.trainY))
weights_DTW = (1 / distances_DTW / sum(1 / distances_DTW)).to_dict()
allocation_DTW = pd.DataFrame({'Component': list(weights_DTW.keys()),
                               'DTWweight(%)': np.multiply(list(weights_DTW.values()), 100)}).sort_values('DTWweight(%)', ascending=False)
allocation_DTW.set_index('Component', inplace=True)
allocation_DTW.reset_index(inplace=True)

# DTW optimization evaluation
stage.testY['DTW'] = stage.testX.dot(list(weights_DTW.values()))
stage.evaluate(stage.testY, 'DTW')

print(allocation_DTW)
print("-" * 50)
print("Leverage Factor: none")
