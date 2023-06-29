import results as res
import pandas as pd
from tensorflow.keras.models import Sequential  # ignore error
from tensorflow.keras.layers import Dense  # ignore error
from sklearn.metrics import precision_score
import time

ml_train_TE = res.TErrors[
    (res.TErrors.index >= pd.to_datetime('2020-09-01')) & (res.TErrors.index <= pd.to_datetime('2022-12-31'))]
ml_test_TE = res.TErrors[
    (res.TErrors.index >= pd.to_datetime('2023-01-01')) & (res.TErrors.index <= pd.to_datetime('2023-05-31'))]
epochs = 20

# define the keras model
dl_model = Sequential()
dl_model.add(Dense(25, input_shape=(5,), activation='relu'))
dl_model.add(Dense(5, activation='relu'))
dl_model.add(Dense(1, activation='sigmoid'))
dl_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
t1 = time.time()
dl_model.fit(ml_train_TE[['NNLS', 'PCRR', 'DTW', 'NNMF', 'PSO']], ml_train_TE['Replica'], epochs=epochs, batch_size=10)
t2 = time.time()
print("-" * 100)
print('Elapsed seconds per epoch: {:.5f}'.format((t2 - t1) / epochs))
dl_model.summary()

# make class predictions with the model
dl_preds = pd.DataFrame()
dl_preds['Ground Truth'] = ml_test_TE[['Replica']]
dl_preds['Predictions'] = (dl_model.predict(ml_test_TE[['NNLS', 'PCRR', 'DTW', 'NNMF', 'PSO']]) > 0.5).astype(bool)
print(dl_preds)
print("-" * 100)
print('DL accuracy score:', precision_score(dl_preds['Ground Truth'], dl_preds['Predictions']))
