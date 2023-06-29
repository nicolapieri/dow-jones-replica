import results as res
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# Replica approximation with ML Random Forest Classifier
print(res.TErrors)
print("-" * 100)

ml_train_TE = res.TErrors[(res.TErrors.index >= pd.to_datetime('2020-09-01')) & (res.TErrors.index <= pd.to_datetime('2022-12-31'))]
ml_test_TE = res.TErrors[(res.TErrors.index >= pd.to_datetime('2023-01-01')) & (res.TErrors.index <= pd.to_datetime('2023-05-31'))]

ml_model = RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=1)

# NNLS
NNLS_fit = ml_model.fit(ml_train_TE[['NNLS']], ml_train_TE['Replica'])
NNLS_preds = pd.Series(NNLS_fit.predict(ml_test_TE[['NNLS']]), index=ml_test_TE.index)
NNLS_ps = precision_score(ml_test_TE['Replica'], NNLS_preds)
print(f'NNLS precision score:\n{NNLS_ps}')
# PCRR
PCRR_fit = ml_model.fit(ml_train_TE[['PCRR']], ml_train_TE['Replica'])
PCRR_preds = pd.Series(PCRR_fit.predict(ml_test_TE[['PCRR']]), index=ml_test_TE.index)
PCRR_ps = precision_score(ml_test_TE['Replica'], PCRR_preds)
print(f'PCRR precision score:\n{PCRR_ps}')
# DTW
DTW_fit = ml_model.fit(ml_train_TE[['DTW']], ml_train_TE['Replica'])
DTW_preds = pd.Series(DTW_fit.predict(ml_test_TE[['DTW']]), index=ml_test_TE.index)
DTW_ps = precision_score(ml_test_TE['Replica'], DTW_preds)
print(f'DTW precision score:\n{DTW_ps}')
# NNMF
NNMF_fit = ml_model.fit(ml_train_TE[['NNMF']], ml_train_TE['Replica'])
NNMF_preds = pd.Series(NNMF_fit.predict(ml_test_TE[['NNMF']]), index=ml_test_TE.index)
NNMF_ps = precision_score(ml_test_TE['Replica'], NNMF_preds)
print(f'NNMF precision score:\n{NNMF_ps}')
# PSO
PSO_fit = ml_model.fit(ml_train_TE[['PSO']], ml_train_TE['Replica'])
PSO_preds = pd.Series(PSO_fit.predict(ml_test_TE[['PSO']]), index=ml_test_TE.index)
PSO_ps = precision_score(ml_test_TE['Replica'], PSO_preds)
print(f'PSO precision score:\n{PSO_ps}')
