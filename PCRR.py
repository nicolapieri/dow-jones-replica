import staging as stage
import pandas as pd
import numpy as np
import pingouin as pg  # necessary to use pcorr() as a pandas method

# portfolio allocation with Partial Correlation (PCRR)
correls = pd.DataFrame({'Correlation': stage.train.corr()['^DJI'],
                        'Partial Correlation': stage.train.pcorr()['^DJI']})
leverage = sum(correls['Partial Correlation'].apply(lambda x: max(0, x))[:-1])
weights = (correls['Partial Correlation'].apply(lambda x: max(0, x))[:-1] / leverage).to_dict()
allocation_PCRR = pd.DataFrame({'Component': list(weights.keys()),
                                'PCRRweight(%)': np.multiply(list(weights.values()), 100)}).sort_values('PCRRweight(%)', ascending=False)
allocation_PCRR.set_index('Component', inplace=True)
allocation_PCRR.reset_index(inplace=True)

# PCRR optimization evaluation
stage.testY['PCRR'] = leverage * stage.testX.dot(list(weights.values()))
stage.evaluate(stage.testY, 'PCRR')
print("\nLeverage Factor:", leverage)
