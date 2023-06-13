import staging as stage
import pandas as pd
import numpy as np
import pingouin as pg  # necessary to use pcorr() as a pandas method

# portfolio allocation with Partial Correlation (PCRR)
correls = pd.DataFrame({'Correlation': stage.train.corr()['^DJI'],
                        'Partial Correlation': stage.train.pcorr()['^DJI']})
leverage = sum(correls['Partial Correlation'].apply(lambda x: max(0, x))[:-1])
weights = (correls['Partial Correlation'].apply(lambda x: max(0, x))[:-1] / leverage).to_dict()
allocation = pd.DataFrame({'Component': list(weights.keys()),
                           'Weight(%)': np.multiply(list(weights.values()), 100)}).sort_values('Weight(%)', ascending=False)
allocation.set_index('Component', inplace=True)
allocation.reset_index(inplace=True)
print("*" * 10, "Partial Correlation (PCRR) Optimization", "*" * 10)
print("\nPortfolio Allocation:")
print(allocation)
print("\nLeverage Factor:", leverage, "\n")
pass
# portfolio optimization evaluation
stage.valY['PCRR'] = leverage * stage.valX.dot(list(weights.values()))
stage.evaluate(stage.valY, 'PCRR')
stage.testY['PCRR'] = leverage * stage.testX.dot(list(weights.values()))
stage.evaluate(stage.testY, 'PCRR')