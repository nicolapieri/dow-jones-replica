import yfinance as yf
import pandas as pd
import numpy as np
# NNLS
from scipy.optimize import nnls
# PCRR (necessary for partial corr)
import pingouin as pg
# DTW
from tslearn.metrics import dtw
# NNMF
from sklearn.decomposition import NMF
# PSO
from pyswarms.utils.search.grid_search import GridSearch
from pyswarms.single.global_best import GlobalBestPSO
# personal packages
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from processing import processing
from learning import learning


def test_opt(composition, change_date, start_date, end_date):
    # creating adjusted closes dataframe
    stocks = pd.DataFrame()
    for ticker in composition:
        globals()[ticker] = pd.DataFrame(yf.download(ticker, start=start_date, end=end_date))
        stocks[ticker] = globals()[ticker].loc[:, 'Adj Close']
    X = stocks.dropna(axis=1, how='any')
    Y = pd.DataFrame()
    Y['^DJI'] = pd.DataFrame(yf.download("^DJI", start=start_date, end=end_date))['Adj Close']

    # portfolio allocation with Non-Negative Least Squares (NNLS)
    NNLS_residual = nnls(X, Y['^DJI'])
    NNLS_leverage = sum(NNLS_residual[0])
    NNLS_weights = dict(zip(X.columns, NNLS_residual[0] / NNLS_leverage))
    NNLS_allocation = pd.DataFrame({'Component': list(NNLS_weights.keys()),
                                    'NNLS-30wg(%)': np.multiply(list(NNLS_weights.values()), 100)}).sort_values(
        'NNLS-30wg(%)', ascending=False)
    NNLS_allocation.set_index('Component', inplace=True)
    NNLS_allocation.reset_index(inplace=True)

    # portfolio allocation with Partial Correlation (PCRR)
    PCRR_correls = pd.DataFrame({'corr': pd.merge(X, Y, on='Date').corr()['^DJI'],
                                 'pcorr': pd.merge(X, Y, on='Date').pcorr()['^DJI']})
    PCRR_leverage = sum(PCRR_correls['pcorr'].drop('^DJI'))
    PCRR_weights = (PCRR_correls['pcorr'].drop('^DJI') / PCRR_leverage).to_dict()
    PCRR_allocation = pd.DataFrame({'Component': list(PCRR_weights.keys()),
                                    'PCRR-30wg(%)': np.multiply(list(PCRR_weights.values()), 100)}).sort_values(
        'PCRR-30wg(%)', ascending=False)
    PCRR_allocation.set_index('Component', inplace=True)
    PCRR_allocation.reset_index(inplace=True)

    # portfolio allocation with Dynamic Time Warping (DTW)
    DTW_distances = X.apply(lambda x: dtw(x, Y))
    DTW_weights = (1 / DTW_distances / sum(1 / DTW_distances)).to_dict()
    DTW_allocation = pd.DataFrame({'Component': list(DTW_weights.keys()),
                                   'DTW-30wg(%)': np.multiply(list(DTW_weights.values()), 100)}).sort_values(
        'DTW-30wg(%)', ascending=False)
    DTW_allocation.set_index('Component', inplace=True)
    DTW_allocation.reset_index(inplace=True)

    # portfolio allocation with Non-Negative Matrix Factorization (NNMF)
    NNMF_coeffs = NMF(n_components=1).fit(X).components_.tolist()[0]
    NNMF_weights = dict(zip(list(X.columns), np.divide(NNMF_coeffs, sum(NNMF_coeffs))))
    NNMF_allocation = pd.DataFrame({'Component': list(NNMF_weights.keys()),
                                    'NNMF-30wg(%)': np.multiply(list(NNMF_weights.values()), 100)}).sort_values(
        'NNMF-30wg(%)', ascending=False)
    NNMF_allocation.set_index('Component', inplace=True)
    NNMF_allocation.reset_index(inplace=True)

    # portfolio allocation with Particle Swarm (PSO)
    def train_particle_loss(coeffs):
        benchmark_tracking_error = np.std(X.dot(coeffs) - Y['^DJI'])
        return benchmark_tracking_error

    def train_swarm(x):
        n_particles = x.shape[0]
        particle_loss = [train_particle_loss(x[i]) for i in range(n_particles)]
        return particle_loss

    g = GridSearch(GlobalBestPSO,
                   objective_func=train_swarm,
                   n_particles=100,
                   dimensions=len(X.columns),
                   options={'c1': [1.5, 2.5], 'c2': [1, 2], 'w': [0.4, 0.5]},
                   bounds=(len(X.columns) * [0],
                           len(X.columns) * [1]),
                   iters=5)

    best_cost, best_pos = g.search()  # Given a matrix of position options search for best position

    optimizer = GlobalBestPSO(n_particles=1000,
                              dimensions=len(X.columns),
                              options=best_pos,
                              bounds=(len(X.columns) * [0],
                                      len(X.columns) * [1]))

    cost, pos = optimizer.optimize(train_swarm, iters=5)  # Given the best position option optimize the cost

    PSO_leverage = sum(pos)
    PSO_weights = dict(zip(list(X.columns), list(pos / PSO_leverage)))
    PSO_allocation = pd.DataFrame({'Component': X.columns,
                                   'PSO-30wg(%)': np.multiply(list(PSO_weights.values()), 100)}).sort_values(
        'PSO-30wg(%)', ascending=False)
    PSO_allocation.set_index('Component', inplace=True)
    PSO_allocation.reset_index(inplace=True)

    if change_date == '2003-01-27':
        global Opt_2003_01_27
        Opt_2003_01_27 = pd.DataFrame()
        Opt_2003_01_27['^DJI'] = Y['^DJI']
        Opt_2003_01_27['NNLS'] = NNLS_leverage * X.dot(list(NNLS_weights.values()))
        Opt_2003_01_27['PCRR'] = PCRR_leverage * X.dot(list(PCRR_weights.values()))
        Opt_2003_01_27['DTW'] = X.dot(list(DTW_weights.values()))
        Opt_2003_01_27['NNMF'] = X.dot(list(NNMF_weights.values()))
        Opt_2003_01_27['PSO'] = PSO_leverage * X.dot(list(PSO_weights.values()))

    if change_date == '2004-04-08':
        global Opt_2004_04_08
        Opt_2004_04_08 = pd.DataFrame()
        Opt_2004_04_08['^DJI'] = Y['^DJI']
        Opt_2004_04_08['NNLS'] = NNLS_leverage * X.dot(list(NNLS_weights.values()))
        Opt_2004_04_08['PCRR'] = PCRR_leverage * X.dot(list(PCRR_weights.values()))
        Opt_2004_04_08['DTW'] = X.dot(list(DTW_weights.values()))
        Opt_2004_04_08['NNMF'] = X.dot(list(NNMF_weights.values()))
        Opt_2004_04_08['PSO'] = PSO_leverage * X.dot(list(PSO_weights.values()))

    if change_date == '2005-11-21':
        global Opt_2005_11_21
        Opt_2005_11_21 = pd.DataFrame()
        Opt_2005_11_21['^DJI'] = Y['^DJI']
        Opt_2005_11_21['NNLS'] = NNLS_leverage * X.dot(list(NNLS_weights.values()))
        Opt_2005_11_21['PCRR'] = PCRR_leverage * X.dot(list(PCRR_weights.values()))
        Opt_2005_11_21['DTW'] = X.dot(list(DTW_weights.values()))
        Opt_2005_11_21['NNMF'] = X.dot(list(NNMF_weights.values()))
        Opt_2005_11_21['PSO'] = PSO_leverage * X.dot(list(PSO_weights.values()))

    if change_date == '2008-02-19':
        global Opt_2008_02_19
        Opt_2008_02_19 = pd.DataFrame()
        Opt_2008_02_19['^DJI'] = Y['^DJI']
        Opt_2008_02_19['NNLS'] = NNLS_leverage * X.dot(list(NNLS_weights.values()))
        Opt_2008_02_19['PCRR'] = PCRR_leverage * X.dot(list(PCRR_weights.values()))
        Opt_2008_02_19['DTW'] = X.dot(list(DTW_weights.values()))
        Opt_2008_02_19['NNMF'] = X.dot(list(NNMF_weights.values()))
        Opt_2008_02_19['PSO'] = PSO_leverage * X.dot(list(PSO_weights.values()))

    if change_date == '2008-09-22':
        global Opt_2008_09_22
        Opt_2008_09_22 = pd.DataFrame()
        Opt_2008_09_22['^DJI'] = Y['^DJI']
        Opt_2008_09_22['NNLS'] = NNLS_leverage * X.dot(list(NNLS_weights.values()))
        Opt_2008_09_22['PCRR'] = PCRR_leverage * X.dot(list(PCRR_weights.values()))
        Opt_2008_09_22['DTW'] = X.dot(list(DTW_weights.values()))
        Opt_2008_09_22['NNMF'] = X.dot(list(NNMF_weights.values()))
        Opt_2008_09_22['PSO'] = PSO_leverage * X.dot(list(PSO_weights.values()))

    if change_date == '2009-06-08':
        global Opt_2009_06_08
        Opt_2009_06_08 = pd.DataFrame()
        Opt_2009_06_08['^DJI'] = Y['^DJI']
        Opt_2009_06_08['NNLS'] = NNLS_leverage * X.dot(list(NNLS_weights.values()))
        Opt_2009_06_08['PCRR'] = PCRR_leverage * X.dot(list(PCRR_weights.values()))
        Opt_2009_06_08['DTW'] = X.dot(list(DTW_weights.values()))
        Opt_2009_06_08['NNMF'] = X.dot(list(NNMF_weights.values()))
        Opt_2009_06_08['PSO'] = PSO_leverage * X.dot(list(PSO_weights.values()))

    if change_date == '2012-09-24':
        global Opt_2012_09_24
        Opt_2012_09_24 = pd.DataFrame()
        Opt_2012_09_24['^DJI'] = Y['^DJI']
        Opt_2012_09_24['NNLS'] = NNLS_leverage * X.dot(list(NNLS_weights.values()))
        Opt_2012_09_24['PCRR'] = PCRR_leverage * X.dot(list(PCRR_weights.values()))
        Opt_2012_09_24['DTW'] = X.dot(list(DTW_weights.values()))
        Opt_2012_09_24['NNMF'] = X.dot(list(NNMF_weights.values()))
        Opt_2012_09_24['PSO'] = PSO_leverage * X.dot(list(PSO_weights.values()))

    if change_date == '2013-09-23':
        global Opt_2013_09_23
        Opt_2013_09_23 = pd.DataFrame()
        Opt_2013_09_23['^DJI'] = Y['^DJI']
        Opt_2013_09_23['NNLS'] = NNLS_leverage * X.dot(list(NNLS_weights.values()))
        Opt_2013_09_23['PCRR'] = PCRR_leverage * X.dot(list(PCRR_weights.values()))
        Opt_2013_09_23['DTW'] = X.dot(list(DTW_weights.values()))
        Opt_2013_09_23['NNMF'] = X.dot(list(NNMF_weights.values()))
        Opt_2013_09_23['PSO'] = PSO_leverage * X.dot(list(PSO_weights.values()))

    if change_date == '2015-03-19':
        global Opt_2015_03_19
        Opt_2015_03_19 = pd.DataFrame()
        Opt_2015_03_19['^DJI'] = Y['^DJI']
        Opt_2015_03_19['NNLS'] = NNLS_leverage * X.dot(list(NNLS_weights.values()))
        Opt_2015_03_19['PCRR'] = PCRR_leverage * X.dot(list(PCRR_weights.values()))
        Opt_2015_03_19['DTW'] = X.dot(list(DTW_weights.values()))
        Opt_2015_03_19['NNMF'] = X.dot(list(NNMF_weights.values()))
        Opt_2015_03_19['PSO'] = PSO_leverage * X.dot(list(PSO_weights.values()))

    if change_date == '2018-06-26':
        global Opt_2018_06_26
        Opt_2018_06_26 = pd.DataFrame()
        Opt_2018_06_26['^DJI'] = Y['^DJI']
        Opt_2018_06_26['NNLS'] = NNLS_leverage * X.dot(list(NNLS_weights.values()))
        Opt_2018_06_26['PCRR'] = PCRR_leverage * X.dot(list(PCRR_weights.values()))
        Opt_2018_06_26['DTW'] = X.dot(list(DTW_weights.values()))
        Opt_2018_06_26['NNMF'] = X.dot(list(NNMF_weights.values()))
        Opt_2018_06_26['PSO'] = PSO_leverage * X.dot(list(PSO_weights.values()))

    if change_date == '2019-04-02':
        global Opt_2019_04_02
        Opt_2019_04_02 = pd.DataFrame()
        Opt_2019_04_02['^DJI'] = Y['^DJI']
        Opt_2019_04_02['NNLS'] = NNLS_leverage * X.dot(list(NNLS_weights.values()))
        Opt_2019_04_02['PCRR'] = PCRR_leverage * X.dot(list(PCRR_weights.values()))
        Opt_2019_04_02['DTW'] = X.dot(list(DTW_weights.values()))
        Opt_2019_04_02['NNMF'] = X.dot(list(NNMF_weights.values()))
        Opt_2019_04_02['PSO'] = PSO_leverage * X.dot(list(PSO_weights.values()))

    if change_date == '2020-04-06':
        global Opt_2020_04_06
        Opt_2020_04_06 = pd.DataFrame()
        Opt_2020_04_06['^DJI'] = Y['^DJI']
        Opt_2020_04_06['NNLS'] = NNLS_leverage * X.dot(list(NNLS_weights.values()))
        Opt_2020_04_06['PCRR'] = PCRR_leverage * X.dot(list(PCRR_weights.values()))
        Opt_2020_04_06['DTW'] = X.dot(list(DTW_weights.values()))
        Opt_2020_04_06['NNMF'] = X.dot(list(NNMF_weights.values()))
        Opt_2020_04_06['PSO'] = PSO_leverage * X.dot(list(PSO_weights.values()))

    if change_date == '2020-08-31':
        global Opt_2020_08_31
        Opt_2020_08_31 = pd.DataFrame()
        Opt_2020_08_31['^DJI'] = Y['^DJI']
        Opt_2020_08_31['NNLS'] = NNLS_leverage * X.dot(list(NNLS_weights.values()))
        Opt_2020_08_31['PCRR'] = PCRR_leverage * X.dot(list(PCRR_weights.values()))
        Opt_2020_08_31['DTW'] = X.dot(list(DTW_weights.values()))
        Opt_2020_08_31['NNMF'] = X.dot(list(NNMF_weights.values()))
        Opt_2020_08_31['PSO'] = PSO_leverage * X.dot(list(PSO_weights.values()))

    else:
        None


def choose_predictors():
    print("Select a couple of optimizers as predictors (NNLS, PCRR, DTW, NNMF, PSO)")
    choice_1 = str(input('Your first choice is: '))
    choice_2 = str(input('Your second choice is: '))
    if choice_1 and choice_2 in ['NNLS', 'PCRR', 'DTW', 'NNMF', 'PSO']:
        if choice_1 != choice_2:
            choices = [choice_1, choice_2]
            return choices
        else:
            print("Cannot select same optimizers.")
            return choose_predictors()
    else:
        print("Invalid input. Please try again.")
        return choose_predictors()
