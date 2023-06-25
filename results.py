import staging as stage
import matplotlib.pyplot as plt
import recon
import NNLS
import PCRR
import DTW
import NNMF
import PSO
import pandas as pd

# creating final results dataframes
results_bystocks = pd.merge(
    pd.merge(
        pd.merge(
            pd.merge(
                pd.merge(
                    pd.merge(
                        pd.merge(
                            pd.merge(
                                pd.merge(
                                    pd.merge(NNLS.NNLStrain_allocation, NNLS.NNLSval_allocation,
                                             on='Component', how='outer'),
                                    PCRR.PCRRtrain_allocation,
                                    on='Component', how='outer'),
                                PCRR.PCRRval_allocation,
                                on='Component', how='outer'),
                            DTW.DTWtrain_allocation,
                            on='Component', how='outer'),
                        DTW.DTWval_allocation,
                        on='Component', how='outer'),
                    NNMF.NNMFtrain_allocation,
                    on='Component', how='outer'),
                NNMF.NNMFval_allocation,
                on='Component', how='outer'),
            PSO.PSOtrain_allocation,
            on='Component', how='outer'),
        PSO.PSOval_allocation,
        on='Component', how='outer'),
    recon.reconstruction,
    on='Component', how='outer')

results_byportfolio = pd.merge(pd.DataFrame(stage.opt_terrors.items(), columns=['Opt_Method', 'Std_Tracking_Errors']),
                               pd.DataFrame(stage.opt_betas.items(), columns=['Opt_Method', 'Portfolio_Beta']),
                               on='Opt_Method').sort_values('Std_Tracking_Errors', ascending=True)

# showing results
plt.style.use('default')
graph = pd.DataFrame()
graph['recon'] = round((stage.testY['recon'].pct_change().dropna() - stage.testY['^DJI'].pct_change().dropna()) * 10000,
                       4)
graph['NNLS'] = round((stage.testY['NNLS'].pct_change().dropna() - stage.testY['^DJI'].pct_change().dropna()) * 10000,
                      4)
graph['PCRR'] = round((stage.testY['PCRR'].pct_change().dropna() - stage.testY['^DJI'].pct_change().dropna()) * 10000,
                      4)
graph['DTW'] = round((stage.testY['DTW'].pct_change().dropna() - stage.testY['^DJI'].pct_change().dropna()) * 10000, 4)
graph['NNMF'] = round((stage.testY['NNMF'].pct_change().dropna() - stage.testY['^DJI'].pct_change().dropna()) * 10000,
                      4)
graph['PSO'] = round((stage.testY['PSO'].pct_change().dropna() - stage.testY['^DJI'].pct_change().dropna()) * 10000, 4)
graph.plot()
plt.title(f"Tracking Errors Summary (bps)")
plt.show()

print("-" * 100)
print(results_byportfolio)
print("-" * 100)
print(results_bystocks)
