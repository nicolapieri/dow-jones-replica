import staging as stage
import pandas as pd
import numpy as np
import pingouin as pg  # necessary to use pcorr() as a pandas method

# portfolio allocation with Partial Correlation (PCRR)
PCRR_correls = pd.DataFrame({'corr': pd.merge(stage.X, stage.Y, on='Date').corr()['^DJI'],
                             'pcorr': pd.merge(stage.X, stage.Y, on='Date').pcorr()['^DJI']})
PCRR_leverage = sum(PCRR_correls['pcorr'].drop('^DJI'))
PCRR_weights = (PCRR_correls['pcorr'].drop('^DJI') / PCRR_leverage).to_dict()
PCRR_allocation = pd.DataFrame({'Component': list(PCRR_weights.keys()),
                                'PCRR-30wg(%)': np.multiply(list(PCRR_weights.values()), 100)}).sort_values(
    'PCRR-30wg(%)', ascending=False)
PCRR_allocation.set_index('Component', inplace=True)
PCRR_allocation.reset_index(inplace=True)

# PCRR portfolio optimization evaluation
stage.Opt['PCRR'] = PCRR_leverage * stage.X.dot(list(PCRR_weights.values()))
stage.evaluate(stage.Opt, 'PCRR')
