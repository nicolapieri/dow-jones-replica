import staging as stage
import math
import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
import pingouin as pg
import matplotlib.pyplot as plt
import pyswarms as ps
from scipy.optimize import nnls
from datetime import date
from sklearn.metrics import mean_squared_error
from IPython.display import display, HTML
from sklearn.decomposition import NMF
from tslearn.metrics import dtw
from pyswarms.utils.search.grid_search import GridSearch
from pyswarms.utils.search.random_search import RandomSearch
from pyswarms.single.global_best import GlobalBestPSO
from pyswarms.utils.plotters import plot_cost_history, plot_contour, plot_surface


def get_portfolio_allocation(trainX, trainY):
    result = nnls(trainX, trainY['^DJI'])
    print('NNLS Residual', round(result[1], 5))

    leverage_factor = sum(result[0])
    weights = result[0] / leverage_factor
    weights = dict(zip(trainX.columns, weights))

    s1 = str(round(leverage_factor, 5)) + "*("
    for component in weights.keys():
        s1 += str(round(weights[component], 5)) + '*' + component + " + "
    s1 = s1[:-3] + ")"

    print("\nPortfolio Allocation:")
    allocation = pd.DataFrame(
        {'Component': list(weights.keys()), 'Weight(%)': np.multiply(list(weights.values()), 100)}).sort_values(
        'Weight(%)', ascending=False)
    allocation.set_index('Component', inplace=True)
    allocation.reset_index(inplace=True)
    display(allocation)

    print('\nPortfolio Simulated: ')
    print(s1)
    print("\nLeverage Factor:", leverage_factor)
    return leverage_factor, weights


# portfolio allocation with Non-Negative Least Squares (NNLS)
leverage_factor, weights = get_portfolio_allocation(stage.trainX, stage.trainY)
sns.heatmap(stage.trainX.corr(), cmap="Purples", vmin=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()
stage.valY['NNLS'] = leverage_factor * stage.valX.dot(list(weights.values()))
stage.evaluate(stage.valY, 'NNLS', is_test=False)
stage.testY['avg26'] = stage.Returns['avg26']
stage.testY['NNLS'] = leverage_factor * stage.testX.dot(list(weights.values()))
stage.evaluate(stage.testY, 'NNLS', is_test=True)
pass

# portfolio allocation with Partial Correlation (PCRR)
correls = pd.DataFrame({'Correlation': stage.train.corr()['^DJI'],
                        'Partial Correlation': stage.train.pcorr()['^DJI']})
print(correls)
weights = correls['Partial Correlation'].apply(lambda x: max(0, x))[:-1]
leverage_factor = sum(weights)
weights = (weights / leverage_factor).to_dict()
s1 = str(round(leverage_factor, 5))+"("
for component, weight in weights.items():
    s1 += str(round(weight, 5))+'*'+component+" + "
s1 = s1[:-3]+")"
print("\nPortfolio Allocation:")
allocation = pd.DataFrame({'Component': list(weights.keys()),
                           'Weight(%)': np.multiply(list(weights.values()), 100)}).sort_values('Weight(%)', ascending=False)
allocation.set_index('Component', inplace=True)
allocation.plot.pie(y='Weight(%)', legend=None)
allocation.reset_index(inplace=True)
display(allocation)
print('\nPortfolio Simulated Returns = ')
print(s1)
print("\nLeverage Factor:", leverage_factor)
stage.valY['Partial Correlation Returns'] = leverage_factor*stage.valX.dot(list(weights.values()))
stage.evaluate(stage.valY, 'Partial Correlation Returns', is_test=False)
stage.testY['Benchmark Close'] = stage.Returns['avg26']
stage.testY['Partial Correlation Returns'] = leverage_factor*stage.testX.dot(list(weights.values()))
stage.evaluate(stage.testY, 'Partial Correlation Returns', is_test=True)
pass
