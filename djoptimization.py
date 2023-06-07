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

# declaring components and weights (https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average)
# excluded CRM, DOW, GS and V for missing values
DJIA_components = {'AAPL': 0.0284, 'AMGN': 0.0548,
                   'AXP': 0.0302, 'BA': 0.0336,
                   'CAT': 0.0452, 'CSCO': 0.0096,
                   'CVX': 0.0350, 'DIS': 0.0189,
                   'HD': 0.0627, 'HON': 0.0417,
                   'IBM': 0.0286, 'INTC': 0.0057,
                   'JNJ': 0.0343, 'JPM': 0.0261,
                   'KO': 0.0122, 'MCD': 0.0524,
                   'MMM': 0.0241, 'MRK': 0.0210,
                   'MSFT': 0.0488, 'NKE': 0.0213,
                   'PG': 0.0286, 'TRV': 0.0362,
                   'UNH': 0.1029, 'VZ': 0.0073,
                   'WBA': 0.0079, 'WMT': 0.0294}