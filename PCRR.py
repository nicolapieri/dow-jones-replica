import staging as stage
import pandas as pd
import numpy as np
import pingouin as pg  # necessary to use pcorr() as a pandas method

# portfolio allocation with Partial Correlation (PCRR)
correls_PCRR = pd.DataFrame({'corr': pd.merge(stage.trainX, stage.trainY, on='Date').corr()['^DJI'],
                             'pcorr': pd.merge(stage.trainX, stage.trainY, on='Date').pcorr()['^DJI']})
leverage_PCRR = sum(correls_PCRR['pcorr'].drop('^DJI'))
weights_PCRR = (correls_PCRR['pcorr'].drop('^DJI') / leverage_PCRR).to_dict()
allocation_PCRR = pd.DataFrame({'Component': list(weights_PCRR.keys()),
                                'PCRRweight(%)': np.multiply(list(weights_PCRR.values()), 100)}).sort_values(
    'PCRRweight(%)', ascending=False)
allocation_PCRR.set_index('Component', inplace=True)
allocation_PCRR.reset_index(inplace=True)

# PCRR optimization evaluation
stage.testY['PCRR'] = leverage_PCRR * stage.testX.dot(list(weights_PCRR.values()))
stage.evaluate(stage.testY, 'PCRR')

print(allocation_PCRR)
print("-" * 50)
print("Leverage Factor:", leverage_PCRR)
