import pandas as pd
import matplotlib.pyplot as plt
import func

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


# different main function config (compat, compat date, compat date +1, next compat date)
func.test_opt(compat_2003_01_27, '2003-01-27', '2003-06-01', '2004-04-08')
func.test_opt(compat_2004_04_08, '2004-04-08', '2004-04-09', '2005-11-21')
func.test_opt(compat_2005_11_21, '2005-11-21', '2005-11-22', '2008-02-19')
func.test_opt(compat_2008_02_19, '2008-02-19', '2008-02-20', '2008-09-22')
func.test_opt(compat_2008_09_22, '2008-09-22', '2008-09-23', '2009-06-08')
func.test_opt(compat_2009_06_08, '2009-06-08', '2009-06-09', '2012-09-24')
func.test_opt(compat_2012_09_24, '2012-09-24', '2012-09-25', '2013-09-23')
func.test_opt(compat_2013_09_23, '2013-09-23', '2013-09-24', '2015-03-19')
func.test_opt(compat_2015_03_19, '2015-03-19', '2015-03-20', '2018-06-26')
func.test_opt(compat_2018_06_26, '2018-06-26', '2018-06-27', '2019-04-02')
func.test_opt(compat_2019_04_02, '2019-04-02', '2019-04-03', '2020-04-06')
func.test_opt(compat_2020_04_06, '2020-04-06', '2020-04-07', '2020-08-31')
func.test_opt(compat_2020_08_31, '2020-08-31', '2020-09-01', '2023-05-31')

# aggregate optimization results
Fin = pd.concat(
    [func.Opt_2003_01_27, func.Opt_2004_04_08, func.Opt_2005_11_21,
     func.Opt_2008_02_19, func.Opt_2008_09_22, func.Opt_2009_06_08,
     func.Opt_2012_09_24, func.Opt_2013_09_23, func.Opt_2015_03_19,
     func.Opt_2018_06_26, func.Opt_2019_04_02, func.Opt_2020_04_06,
     func.Opt_2020_08_31])

# computing tracking errors
TE = pd.DataFrame()
TE['NNLS'] = round((Fin['NNLS'].pct_change().dropna() - Fin['^DJI'].pct_change().dropna()) * 10000, 4)
TE['PCRR'] = round((Fin['PCRR'].pct_change().dropna() - Fin['^DJI'].pct_change().dropna()) * 10000, 4)
TE['DTW'] = round((Fin['DTW'].pct_change().dropna() - Fin['^DJI'].pct_change().dropna()) * 10000, 4)
TE['NNMF'] = round((Fin['NNMF'].pct_change().dropna() - Fin['^DJI'].pct_change().dropna()) * 10000, 4)
TE['PSO'] = round((Fin['PSO'].pct_change().dropna() - Fin['^DJI'].pct_change().dropna()) * 10000, 4)
TE[(TE.index >= pd.to_datetime('2023-05-01')) & (TE.index <= pd.to_datetime('2023-05-31'))].plot()
TE['Mean'] = TE.mean(axis=1)
TE['Replica'] = (abs(TE['Mean']) < 5).astype(bool)

# plotting
plt.title(f"Optimizations Tracking Errors (last month only)")
plt.ylabel('Basis Points (bps)')
plt.hlines(y=5, xmin="2023-05-01", xmax="2023-05-31", colors='indigo', linestyle='dashed')
plt.hlines(y=-5, xmin="2023-05-01", xmax="2023-05-31", colors='indigo', linestyle='dashed')
plt.show()
print(TE)
