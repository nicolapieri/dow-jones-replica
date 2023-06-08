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

ds = yv.Returns.drop('avg26', axis=1)

train = ds[(ds.index >= pd.to_datetime("1993-01-01")) & (ds.index <= pd.to_datetime("2021-12-31"))]
val = ds[(ds.index >= pd.to_datetime("2022-01-01")) & (ds.index <= pd.to_datetime("2022-11-30"))]
test = ds[(ds.index >= pd.to_datetime("2022-12-01")) & (ds.index <= pd.to_datetime("2022-12-31"))]

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
pass


def evaluate(df, portfolio_col, is_test):
    index_hpr = (yv.Returns['^DJI'][-1] - yv.Returns['^DJI'][0]) / yv.Returns['^DJI'][0]

    if is_test:
        benchmark_hpr = (yv.Returns['avg26'][-1] - yv.Returns['avg26'][0]) / yv.Returns['avg26'][0]
        benchmark_active_return = benchmark_hpr - index_hpr
        benchmark_tracking_error = np.std(yv.Returns['avg26'] - yv.Returns['^DJI'])
        info_ratio = benchmark_active_return / benchmark_tracking_error
        print("\nBenchmark (avg26)")
        print("*" * 30)
        print("Active Return:", round(benchmark_active_return, 5))
        print("Tracking Error:", round(benchmark_tracking_error * 10000), " bps")
        print("Information Ratio:", round(info_ratio, 5))
        print("Returns RMSE:", mean_squared_error(yv.Returns['avg26'], yv.Returns['^DJI'], squared=False))

    if portfolio_col:
        portfolio_hpr = (df[portfolio_col][-1] - df[portfolio_col][0]) / df[portfolio_col][0]
        portfolio_active_return = portfolio_hpr - index_hpr
        portfolio_tracking_error = np.std(df[portfolio_col] - yv.Returns['^DJI'])
        info_ratio = portfolio_active_return / portfolio_tracking_error
        print("\nTrial Portfolio")
        print("*" * 30)
        print("Active Return:", round(portfolio_active_return, 5))
        print("Tracking Error:", round(portfolio_tracking_error * 10000), "bps")
        print("Information Ratio:", round(info_ratio, 5))
        print("Returns RMSE:", mean_squared_error(df[portfolio_col], df['^DJI'], squared=False))

    df.plot()
    plt.title("Non-negative Least Squares (NNLS) Optimization")
    plt.show()
    pass


valY['NNLS'] = leverage_factor * valX.dot(list(weights.values()))
evaluate(valY, 'NNLS', is_test=False)

testY['avg26'] = yv.Returns['avg26']
testY['NNLS'] = leverage_factor * testX.dot(list(weights.values()))
evaluate(testY, 'NNLS', is_test=True)
