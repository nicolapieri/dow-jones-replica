import results as res
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# Replica approximation with ML Random Forest Classifier
print(res.TErrors)
print("-" * 100)

train_TE = res.TErrors[(res.TErrors.index >= pd.to_datetime('2020-09-01')) & (res.TErrors.index <= pd.to_datetime('2022-12-31'))]
test_TE = res.TErrors[(res.TErrors.index >= pd.to_datetime('2023-01-01')) & (res.TErrors.index <= pd.to_datetime('2023-05-31'))]

model = RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=1)

# NNLS
NNLS_fit = model.fit(train_TE[['NNLS']], train_TE['Replica'])
NNLS_preds = pd.Series(NNLS_fit.predict(test_TE[['NNLS']]), index=test_TE.index)
NNLS_ps = precision_score(test_TE['Replica'], NNLS_preds)
print(f'NNLS precision score:\n{NNLS_ps}')
# PCRR
PCRR_fit = model.fit(train_TE[['PCRR']], train_TE['Replica'])
PCRR_preds = pd.Series(PCRR_fit.predict(test_TE[['PCRR']]), index=test_TE.index)
PCRR_ps = precision_score(test_TE['Replica'], PCRR_preds)
print(f'PCRR precision score:\n{PCRR_ps}')
# DTW
DTW_fit = model.fit(train_TE[['DTW']], train_TE['Replica'])
DTW_preds = pd.Series(DTW_fit.predict(test_TE[['DTW']]), index=test_TE.index)
DTW_ps = precision_score(test_TE['Replica'], DTW_preds)
print(f'DTW precision score:\n{DTW_ps}')
# NNMF
NNMF_fit = model.fit(train_TE[['NNMF']], train_TE['Replica'])
NNMF_preds = pd.Series(NNMF_fit.predict(test_TE[['NNMF']]), index=test_TE.index)
NNMF_ps = precision_score(test_TE['Replica'], NNMF_preds)
print(f'NNMF precision score:\n{NNMF_ps}')
# PSO
PSO_fit = model.fit(train_TE[['PSO']], train_TE['Replica'])
PSO_preds = pd.Series(PSO_fit.predict(test_TE[['PSO']]), index=test_TE.index)
PSO_ps = precision_score(test_TE['Replica'], PSO_preds)
print(f'PSO precision score:\n{PSO_ps}')
