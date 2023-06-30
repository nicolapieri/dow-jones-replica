import staging as stage
import matplotlib.pyplot as plt
import NNLS
import PCRR
import DTW
import NNMF
import PSO
import pandas as pd

# creating final results dataframes
Opt_methods = pd.DataFrame(stage.opt_terrors.items(), columns=['Opt_Method', 'Std_Tracking_Errors']).sort_values(
    'Std_Tracking_Errors', ascending=True)

Opt_portfolios = pd.merge(pd.merge(pd.merge(pd.merge(NNLS.NNLS_allocation, PCRR.PCRR_allocation, on='Component', how='outer'),
                                            DTW.DTW_allocation, on='Component', how='outer'),
                                   NNMF.NNMF_allocation, on='Component', how='outer'),
                          PSO.PSO_allocation, on='Component', how='outer').fillna(0)

# showing results
plt.style.use('default')

TErrors = pd.DataFrame()
TErrors['NNLS'] = round((stage.Opt['NNLS'].pct_change().dropna() - stage.Opt['^DJI'].pct_change().dropna()) * 10000, 4)
TErrors['PCRR'] = round((stage.Opt['PCRR'].pct_change().dropna() - stage.Opt['^DJI'].pct_change().dropna()) * 10000, 4)
TErrors['DTW'] = round((stage.Opt['DTW'].pct_change().dropna() - stage.Opt['^DJI'].pct_change().dropna()) * 10000, 4)
TErrors['NNMF'] = round((stage.Opt['NNMF'].pct_change().dropna() - stage.Opt['^DJI'].pct_change().dropna()) * 10000, 4)
TErrors['PSO'] = round((stage.Opt['PSO'].pct_change().dropna() - stage.Opt['^DJI'].pct_change().dropna()) * 10000, 4)
TErrors[(TErrors.index >= pd.to_datetime('2023-05-01')) & (TErrors.index <= pd.to_datetime('2023-05-31'))].plot()
TErrors['Mean'] = TErrors.mean(axis=1)
TErrors['Replica'] = (abs(TErrors.mean(axis=1)) < 5).astype(bool)

plt.title(f"Portfolios Tracking Errors")
plt.ylabel('Basis Points (bps)')
plt.hlines(y=5, xmin="2023-05-01", xmax="2023-05-31", colors='indigo', linestyle='dashed')
plt.hlines(y=-5, xmin="2023-05-01", xmax="2023-05-31", colors='indigo', linestyle='dashed')
plt.show()

print("-" * 100)
print(Opt_methods)
print("-" * 100)
print(Opt_portfolios)
print("-" * 100)
print(TErrors)
