import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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


# Dow Jones compositions over years
compat_2003_01_27 = ["MMM", "KODK", "JNJ", "AA", "XOM", "JPM", "MO", "GE", "MCD", "AXP",
                     "GM", "MRK", "T", "HPQ", "MSFT", "BA", "HD", "PG", "CAT", "HON",
                     "SBAC", "C", "INTC", "UTX", "KO", "IBM", "WMT", "DD", "IP", "DIS"]

compat_2004_04_08 = ["MMM", "PFE", "JNJ", "AA", "XOM", "JPM", "MO", "GE", "MCD", "AXP",
                     "GM", "MRK", "AIG", "HPQ", "MSFT", "BA", "HD", "PG", "CAT", "HON",
                     "SBAC", "C", "INTC", "UTX", "KO", "IBM", "WMT", "DD", "VZ", "DIS"]

compat_2005_11_21 = ["MMM", "PFE", "JNJ", "AA", "XOM", "JPM", "MO", "GE", "MCD", "AXP",
                     "GM", "MRK", "AIG", "HPQ", "MSFT", "BA", "HD", "PG", "CAT", "HON",
                     "T", "C", "INTC", "UTX", "KO", "IBM", "WMT", "DD", "VZ", "DIS"]

compat_2008_02_19 = ["MMM", "PFE", "JNJ", "AA", "XOM", "JPM", "BAC", "GE", "MCD", "AXP",
                     "GM", "MRK", "AIG", "HPQ", "MSFT", "BA", "HD", "PG", "CAT", "CVX",
                     "T", "C", "INTC", "UTX", "KO", "IBM", "WMT", "DD", "VZ", "DIS"]

compat_2008_09_22 = ["MMM", "PFE", "JNJ", "AA", "XOM", "JPM", "BAC", "GE", "MCD", "AXP",
                     "GM", "MRK", "KHC", "HPQ", "MSFT", "BA", "HD", "PG", "CAT", "CVX",
                     "T", "C", "INTC", "UTX", "KO", "IBM", "WMT", "DD", "VZ", "DIS"]

compat_2009_06_08 = ["MMM", "PFE", "JNJ", "AA", "XOM", "JPM", "BAC", "GE", "MCD", "AXP",
                     "TRV", "MRK", "KHC", "HPQ", "MSFT", "BA", "HD", "PG", "CAT", "CVX",
                     "T", "CSCO", "INTC", "UTX", "KO", "IBM", "WMT", "DD", "VZ", "DIS"]

compat_2012_09_24 = ["MMM", "PFE", "JNJ", "AA", "XOM", "JPM", "BAC", "GE", "MCD", "AXP",
                     "TRV", "MRK", "UNH", "HPQ", "MSFT", "BA", "HD", "PG", "CAT", "CVX",
                     "T", "CSCO", "INTC", "UTX", "KO", "IBM", "WMT", "DD", "VZ", "DIS"]

compat_2013_09_23 = ["MMM", "PFE", "JNJ", "GS", "XOM", "JPM", "NKE", "GE", "MCD", "AXP",
                     "TRV", "MRK", "UNH", "V", "MSFT", "BA", "HD", "PG", "CAT", "CVX",
                     "T", "CSCO", "INTC", "UTX", "KO", "IBM", "WMT", "DD", "VZ", "DIS"]

compat_2015_03_19 = ["MMM", "PFE", "JNJ", "GS", "XOM", "JPM", "NKE", "GE", "MCD", "AXP",
                     "TRV", "MRK", "UNH", "V", "MSFT", "BA", "HD", "PG", "CAT", "CVX",
                     "AAPL", "CSCO", "INTC", "UTX", "KO", "IBM", "WMT", "DD", "VZ", "DIS"]

compat_2018_06_26 = ["MMM", "PFE", "JNJ", "GS", "XOM", "JPM", "NKE", "WBA", "MCD", "AXP",
                     "TRV", "MRK", "UNH", "V", "MSFT", "BA", "HD", "PG", "CAT", "CVX",
                     "AAPL", "CSCO", "INTC", "UTX", "KO", "IBM", "WMT", "DD", "VZ", "DIS"]

compat_2019_04_02 = ["MMM", "PFE", "JNJ", "GS", "XOM", "JPM", "NKE", "WBA", "MCD", "AXP",
                     "TRV", "MRK", "UNH", "V", "MSFT", "BA", "HD", "PG", "CAT", "CVX",
                     "AAPL", "CSCO", "INTC", "UTX", "KO", "IBM", "WMT", "DOW", "VZ", "DIS"]

compat_2020_04_06 = ["MMM", "PFE", "JNJ", "GS", "XOM", "JPM", "NKE", "WBA", "MCD", "AXP",
                     "TRV", "MRK", "UNH", "V", "MSFT", "BA", "HD", "PG", "CAT", "CVX",
                     "AAPL", "CSCO", "INTC", "RTX", "KO", "IBM", "WMT", "DOW", "VZ", "DIS"]

compat_2020_08_31 = ["MMM", "AMGN", "JNJ", "GS", "HON", "JPM", "NKE", "WBA", "MCD", "AXP",
                     "TRV", "MRK", "UNH", "V", "MSFT", "BA", "HD", "PG", "CAT", "CVX",
                     "AAPL", "CSCO", "INTC", "CRM", "KO", "IBM", "WMT", "DOW", "VZ", "DIS"]

# applying optimization methods
test_opt(compat_2003_01_27, '2003-01-27', '2003-06-01', '2004-04-08')
test_opt(compat_2004_04_08, '2004-04-08', '2004-04-09', '2005-11-21')
test_opt(compat_2005_11_21, '2005-11-21', '2005-11-22', '2008-02-19')
test_opt(compat_2008_02_19, '2008-02-19', '2008-02-20', '2008-09-22')
test_opt(compat_2008_09_22, '2008-09-22', '2008-09-23', '2009-06-08')
test_opt(compat_2009_06_08, '2009-06-08', '2009-06-09', '2012-09-24')
test_opt(compat_2012_09_24, '2012-09-24', '2012-09-25', '2013-09-23')
test_opt(compat_2013_09_23, '2013-09-23', '2013-09-24', '2015-03-19')
test_opt(compat_2015_03_19, '2015-03-19', '2015-03-20', '2018-06-26')
test_opt(compat_2018_06_26, '2018-06-26', '2018-06-27', '2019-04-02')
test_opt(compat_2019_04_02, '2019-04-02', '2019-04-03', '2020-04-06')
test_opt(compat_2020_04_06, '2020-04-06', '2020-04-07', '2020-08-31')
test_opt(compat_2020_08_31, '2020-08-31', '2020-09-01', '2023-05-31')

# aggregating results
Fin = pd.concat(
    [Opt_2003_01_27, Opt_2004_04_08, Opt_2005_11_21, Opt_2008_02_19, Opt_2008_09_22, Opt_2009_06_08, Opt_2012_09_24,
     Opt_2013_09_23, Opt_2015_03_19, Opt_2018_06_26, Opt_2019_04_02, Opt_2020_04_06, Opt_2020_08_31])

TE = pd.DataFrame()
TE['NNLS'] = round((Fin['NNLS'].pct_change().dropna() - Fin['^DJI'].pct_change().dropna()) * 10000, 4)
TE['PCRR'] = round((Fin['PCRR'].pct_change().dropna() - Fin['^DJI'].pct_change().dropna()) * 10000, 4)
TE['DTW'] = round((Fin['DTW'].pct_change().dropna() - Fin['^DJI'].pct_change().dropna()) * 10000, 4)
TE['NNMF'] = round((Fin['NNMF'].pct_change().dropna() - Fin['^DJI'].pct_change().dropna()) * 10000, 4)
TE['PSO'] = round((Fin['PSO'].pct_change().dropna() - Fin['^DJI'].pct_change().dropna()) * 10000, 4)
TE[(TE.index >= pd.to_datetime('2023-05-01')) & (TE.index <= pd.to_datetime('2023-05-31'))].plot()
TE['Mean'] = TE.mean(axis=1)
TE['Replica'] = (abs(TE['Mean']) < 5).astype(bool)

plt.title(f"Optimizations Tracking Errors (last month only)")
plt.ylabel('Basis Points (bps)')
plt.hlines(y=5, xmin="2023-05-01", xmax="2023-05-31", colors='indigo', linestyle='dashed')
plt.hlines(y=-5, xmin="2023-05-01", xmax="2023-05-31", colors='indigo', linestyle='dashed')
plt.show()
print(TE)

# train-test split
train_TE = TE[(TE.index >= pd.to_datetime('2003-06-01')) & (TE.index <= pd.to_datetime('2019-05-31'))]
test_TE = TE[(TE.index >= pd.to_datetime('2019-06-01')) & (TE.index <= pd.to_datetime('2023-05-31'))]
epochs = 20

# choosing predictors
print("-" * 100)
#predictors = choose_predictors()
predictors = sys.argv[1:]

# ML modelling
ml_model = RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=1)
ml_model.fit(train_TE[predictors], train_TE['Replica'])

# DL modelling
dl_model = Sequential()
dl_model.add(Dense(4, input_shape=(2,), activation='relu'))
dl_model.add(Dense(2, activation='relu'))
dl_model.add(Dense(1, activation='sigmoid'))
dl_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
t1 = time.time()
dl_model.fit(train_TE[predictors], train_TE['Replica'], epochs=epochs, batch_size=10)
t2 = time.time()
print("-" * 100)
print('Elapsed seconds per epoch: {:.5f}'.format((t2 - t1) / epochs))
dl_model.summary()

# presenting results
replica_preds = pd.DataFrame()
replica_preds['Ground_Truth'] = test_TE[['Replica']]
replica_preds['ML_preds'] = pd.Series(ml_model.predict(test_TE[predictors]), index=test_TE.index)
replica_preds['DL_preds'] = (dl_model.predict(test_TE[predictors]) > 0.5).astype(bool)

print("-" * 100)
print(f'Used the pair of optimizers {predictors} as predictors.')
print("-" * 100)
print(replica_preds)
print("-" * 100)
print('ML accuracy score:', precision_score(replica_preds['Ground_Truth'], replica_preds['ML_preds']))
print("-" * 100)
print('DL accuracy score:', precision_score(replica_preds['Ground_Truth'], replica_preds['DL_preds']))
