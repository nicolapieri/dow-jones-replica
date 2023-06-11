import staging as stage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

df = pd.DataFrame()
df['dow jones'] = stage.Returns['^DJI']
df['portfolio'] = stage.Returns['sel26']

index_hpr = (df['dow jones'][-1] - df['dow jones'][0]) / df['dow jones'][0]
benchmark_hpr = (df['portfolio'][-1] - df['portfolio'][0]) / df['portfolio'][0]
benchmark_active_return = benchmark_hpr - index_hpr
benchmark_tracking_error = np.std(df['portfolio'] - df['dow jones'])
info_ratio = benchmark_active_return / benchmark_tracking_error

print("\nPortfolio (sel26)")
print("*" * 40)
print("Active Return:", round(benchmark_active_return, 5))
print("Tracking Error:", round(benchmark_tracking_error * 10000), " bps")
print("Information Ratio:", round(info_ratio, 5))
print("Returns RMSE:", mean_squared_error(df['portfolio'], df['dow jones'], squared=False))
df.plot()
plt.title("Equally weighted 26 stocks selection")
plt.show()