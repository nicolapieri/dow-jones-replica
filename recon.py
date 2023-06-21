import staging as stage
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# latest dow divisor (2021-11-04)
dow_div = 0.15172752595384

# dow jones reconstruction
betas = {}
price_change = stage.Closes.pct_change().dropna()
for ticker in stage.components:
    betas[ticker] = float(LinearRegression().fit(np.array(price_change[ticker]).reshape((-1, 1)),
                                                 np.array(price_change['^DJI'])).coef_)
reconstruction = pd.DataFrame(betas.items(), columns=['Component', 'Beta']).sort_values('Beta', ascending=False)
reconstruction.set_index('Component', inplace=True)
reconstruction.reset_index(inplace=True)

# recon optimization evaluation
stage.testY['recon'] = stage.testX.sum(axis=1) / dow_div
stage.evaluate(stage.testY, 'recon')

print(reconstruction)
print("-" * 50)
print(f"Dow Divisor: {dow_div}")
