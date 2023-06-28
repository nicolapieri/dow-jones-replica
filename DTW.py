import staging as stage
import pandas as pd
import numpy as np
from tslearn.metrics import dtw

# portfolio allocation with Dynamic Time Warping (DTW)
DTW_distances = stage.X.apply(lambda x: dtw(x, stage.Y))
DTW_weights = (1 / DTW_distances / sum(1 / DTW_distances)).to_dict()
DTW_allocation = pd.DataFrame({'Component': list(DTW_weights.keys()),
                               'DTW-30wg(%)': np.multiply(list(DTW_weights.values()), 100)}).sort_values('DTW-30wg(%)',
                                                                                                         ascending=False)
DTW_allocation.set_index('Component', inplace=True)
DTW_allocation.reset_index(inplace=True)

# DTW portfolio optimization evaluation
stage.Opt['DTW'] = stage.X.dot(list(DTW_weights.values()))
stage.evaluate(stage.Opt, 'DTW')
