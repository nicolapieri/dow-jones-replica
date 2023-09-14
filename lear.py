from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
import pandas as pd
import aggr

# train/test split
train_TE = aggr.TE[(aggr.TE.index >= pd.to_datetime('2003-06-01')) & (aggr.TE.index <= pd.to_datetime('2019-05-31'))]
test_TE = aggr.TE[(aggr.TE.index >= pd.to_datetime('2019-06-01')) & (aggr.TE.index <= pd.to_datetime('2023-05-31'))]
epochs = 20

# predictors = ['NNLS', 'PCRR']
# predictors = ['NNLS','DTW']
# predictors = ['NNLS','NNMF']
# predictors = ['NNLS', 'PSO']
# predictors = ['PCRR', 'DTW']
# predictors = ['PCRR', 'NNMF']
# predictors = ['PCRR', 'PSO']
# predictors = ['DTW', 'NNMF']
# predictors = ['DTW', 'PSO']
predictors = ['NNMF', 'PSO']

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

# present learning results
replica_preds = pd.DataFrame()
replica_preds['Ground_Truth'] = test_TE[['Replica']]
replica_preds['ML_preds'] = pd.Series(ml_model.predict(test_TE[predictors]), index=test_TE.index)
replica_preds['DL_preds'] = (dl_model.predict(test_TE[predictors]) > 0.5).astype(bool)

print("-" * 100)
print(replica_preds)
print("-" * 100)
print('ML accuracy score:', precision_score(replica_preds['Ground_Truth'], replica_preds['ML_preds']))
print("-" * 100)
print('DL accuracy score:', precision_score(replica_preds['Ground_Truth'], replica_preds['DL_preds']))
