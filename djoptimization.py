import yvariables as yv
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

df = yv.Returns

train = df[(df.index >= pd.to_datetime("1993-01-01")) & (df.index <= pd.to_datetime("2002-12-31"))]
val = df[(df.index >= pd.to_datetime("2003-01-01")) & (df.index <= pd.to_datetime("2012-12-31"))]
test = df[(df.index >= pd.to_datetime("2013-01-01")) & (df.index <= pd.to_datetime("2022-12-31"))]

trainX = train.drop('^DJI', axis=1)
trainY = pd.DataFrame({'^DJI': train['^DJI']})
valX = val.drop('^DJI', axis=1)
valY = pd.DataFrame({'^DJI': val['^DJI']})
testX = test.drop('^DJI', axis=1)
testY = pd.DataFrame({'^DJI': test['^DJI']})


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


leverage_factor, weights = get_portfolio_allocation(trainX, trainY)
sns.heatmap(trainX.corr(), cmap="Purples", vmin=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()