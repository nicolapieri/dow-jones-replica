import numpy as np
import staging as yv

# declaring Dow Jones index Beta (DJI_Beta = 0.9792634776113391 since CRM, DOW, GS and V were removed)
DJI_Beta = 1

# computing the Dow Jones stocks Beta
Stocks_Betas = []
for symbol in yv.composition:
    globals()[symbol + '_Beta'] = np.polyfit(yv.Returns['DJI_Returns'],
                                             yv.Returns[f'{symbol}_Returns'], 1)[0]
    Stocks_Betas.append(globals()[symbol + '_Beta'])