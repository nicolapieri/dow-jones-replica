import staging as stage
import pandas as pd
import numpy as np
from tslearn.metrics import dtw

# portfolio allocation with Dynamic Time Warping (DTW)
distances = stage.trainX.apply(lambda x: dtw(x, stage.trainY))
weights = (1 / distances / sum(1 / distances)).to_dict()
allocation = pd.DataFrame({'Component': list(weights.keys()),
                           'Weight(%)': np.multiply(list(weights.values()), 100)}).sort_values('Weight(%)', ascending=False)
allocation.set_index('Component', inplace=True)
allocation.reset_index(inplace=True)
print("*" * 10, "Dynamic Time Warping (DTW) Optimization", "*" * 10)
print("\nPortfolio Allocation:")
print(allocation)
print('\n')
pass

# portfolio optimization evaluation
stage.valY['DTW'] = stage.valX.dot(list(weights.values()))
stage.evaluate(stage.valY, 'DTW')
stage.testY['DTW'] = stage.testX.dot(list(weights.values()))
stage.evaluate(stage.testY, 'DTW')