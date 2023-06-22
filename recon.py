import staging as stage
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# latest dow divisor (2021-11-04)
dow_div = 0.15172752595384

# dow jones reconstruction
betas = {}
for ticker in stage.components:
    betas[ticker] = float(LinearRegression().fit(np.array(stage.testX.pct_change().dropna()[ticker]).reshape((-1, 1)),
                                                 np.array(stage.testY.pct_change().dropna()['^DJI'])).coef_)
reconstruction = pd.DataFrame(betas.items(), columns=['Component', 'Beta_stock']).sort_values('Beta_stock',
                                                                                              ascending=False)
reconstruction.set_index('Component', inplace=True)
reconstruction.reset_index(inplace=True)

# recon optimization evaluation
stage.testY['recon'] = stage.testX.sum(axis=1) / dow_div
stage.evaluate(stage.testY, 'recon')

print(reconstruction)
print("-" * 50)
print(f"Dow Divisor: {dow_div}")
