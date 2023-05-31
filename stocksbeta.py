import numpy as np
import yvariables as yv

# computing the Beta
for symbol in yv.composition:
    globals()[symbol + '_Beta'] = np.polyfit(yv.Returns['DJI_Returns'],
                                             yv.Returns[f'{symbol}_Returns'], 1)[0]

print(MMM_Beta, )
