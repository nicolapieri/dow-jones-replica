import staging as stage
import matplotlib.pyplot as plt
import recon
import NNLS
import PCRR
import DTW
import NNMF
import PSO
import pandas as pd

# creating final results dataframe
results = pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(recon.reconstruction,
                                                       NNLS.allocation_NNLS, on='Component'),
                                              PCRR.allocation_PCRR, on='Component'),
                                     DTW.allocation_DTW, on='Component'),
                            NNMF.allocation_NNMF, on='Component'),
                   PSO.allocation_PSO, on='Component')
# results['AVGweight(%)'] = results.mean(axis=1)

# showing results
graph = pd.DataFrame()
graph['recon'] = round((stage.testY['recon'].pct_change().dropna() - stage.testY['^DJI'].pct_change().dropna()) * 10000, 4)
graph['NNLS'] = round((stage.testY['NNLS'].pct_change().dropna() - stage.testY['^DJI'].pct_change().dropna()) * 10000, 4)
graph['PCRR'] = round((stage.testY['PCRR'].pct_change().dropna() - stage.testY['^DJI'].pct_change().dropna()) * 10000, 4)
graph['DTW'] = round((stage.testY['DTW'].pct_change().dropna() - stage.testY['^DJI'].pct_change().dropna()) * 10000, 4)
graph['NNMF'] = round((stage.testY['NNMF'].pct_change().dropna() - stage.testY['^DJI'].pct_change().dropna()) * 10000, 4)
graph['PSO'] = round((stage.testY['PSO'].pct_change().dropna() - stage.testY['^DJI'].pct_change().dropna()) * 10000, 4)
graph.plot()
plt.title(f"Daily Deviation Summary (bps)")
plt.show()
print("-" * 100)
print('\n Average Tracking Errors:', stage.opt_performances)
print("-" * 100)
print(results)
