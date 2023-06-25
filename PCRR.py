import staging as stage
import pandas as pd
import numpy as np
import pingouin as pg  # necessary to use pcorr() as a pandas method

# portfolio allocation with Partial Correlation (PCRR)
PCRRtrain_correls = pd.DataFrame({'corr': pd.merge(stage.trainX, stage.trainY, on='Date').corr()['^DJI'],
                                  'pcorr': pd.merge(stage.trainX, stage.trainY, on='Date').pcorr()['^DJI']})
PCRRtrain_leverage = sum(PCRRtrain_correls['pcorr'].drop('^DJI'))
PCRRtrain_weights = (PCRRtrain_correls['pcorr'].drop('^DJI') / PCRRtrain_leverage).to_dict()
PCRRtrain_allocation = pd.DataFrame({'Component': list(PCRRtrain_weights.keys()),
                                     'PCRR-30wg(%)': np.multiply(list(PCRRtrain_weights.values()), 100)}).sort_values(
    'PCRR-30wg(%)', ascending=False)
PCRRtrain_allocation.set_index('Component', inplace=True)
PCRRtrain_allocation.reset_index(inplace=True)

# validation with top 10 weighted stocks
PCRRval_correls = pd.DataFrame(
    {'corr': pd.merge(stage.valX[PCRRtrain_allocation['Component'][0:10]], stage.valY, on='Date').corr()['^DJI'],
     'pcorr': pd.merge(stage.valX[PCRRtrain_allocation['Component'][0:10]], stage.valY, on='Date').pcorr()['^DJI']})
PCRRval_leverage = sum(PCRRval_correls['pcorr'].drop('^DJI'))
PCRRval_weights = (PCRRval_correls['pcorr'].drop('^DJI') / PCRRval_leverage).to_dict()
PCRRval_allocation = pd.DataFrame({'Component': list(PCRRval_weights.keys()),
                                   'PCRR-10wg(%)': np.multiply(list(PCRRval_weights.values()), 100)}).sort_values(
    'PCRR-10wg(%)', ascending=False)
PCRRval_allocation.set_index('Component', inplace=True)
PCRRval_allocation.reset_index(inplace=True)

# PCRR portfolio optimization evaluation
stage.testY['PCRR'] = PCRRval_leverage * stage.testX[PCRRval_allocation['Component'][0:10]].dot(list(PCRRval_weights.values()))
stage.evaluate(stage.testY, 'PCRR')

print(PCRRval_allocation)
print("-" * 50)
print("Leverage Factor:", PCRRval_leverage)
