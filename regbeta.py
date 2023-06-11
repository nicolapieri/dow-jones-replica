import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import staging as yv


def beta_regression(symbol):
    const = np.polyfit(yv.Returns['DJI_Returns'], yv.Returns[f'{symbol}_Returns'], 1)[1]
    MMM_Beta = np.polyfit(yv.Returns['DJI_Returns'], yv.Returns[f'{symbol}_Returns'], 1)[0]

    model = sm.OLS(yv.Returns[f'{symbol}_Returns'],
                   sm.add_constant(yv.Returns['DJI_Returns'])).fit()
    print(model.summary())
    print('MMM Beta:', '{:.4f}'.format(MMM_Beta))

    plt.scatter(yv.Returns['DJI_Returns'], yv.Returns[f'{symbol}_Returns'], color='pink')
    plt.plot(yv.Returns['DJI_Returns'], MMM_Beta * yv.Returns['DJI_Returns'] + const, color='purple')
    plt.xlabel('DJI_Returns')
    plt.ylabel(f'{symbol}_Returns')
    plt.text(60, 60, f'y = {const} + {MMM_Beta} * DJI_Returns', fontsize=22, color='red')
    plt.show()


beta_regression('MMM')
