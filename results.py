import staging as stage
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

print('\nTracking Errors:', stage.opt_performances)
print("-" * 70)
print(results)
