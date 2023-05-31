import numpy as np
import yvariables as yv

# computing the Beta
Stocks_Betas = []
for symbol in yv.composition:
    globals()[symbol + '_Beta'] = np.polyfit(yv.Returns['DJI_Returns'],
                                             yv.Returns[f'{symbol}_Returns'], 1)[0]
    Stocks_Betas.append(globals()[symbol + '_Beta'])

DJI_Beta = sum(Stocks_Betas) / len(Stocks_Betas)
print(DJI_Beta)