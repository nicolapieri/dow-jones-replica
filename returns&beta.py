import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# reading yahoo datasets
DJI = pd.read_csv(".\\ydatasets\\^DJI.csv")
MMM = pd.read_csv(".\\ydatasets\\MMM.csv")

# creating returns dataframe
Returns = pd.DataFrame()
Returns['DJI_Returns'] = (DJI['Adj Close'] - DJI['Adj Close'].shift(1)) / DJI['Adj Close'].shift(1)
Returns['MMM_Returns'] = (MMM['Adj Close'] - MMM['Adj Close'].shift(1)) / MMM['Adj Close'].shift(1)
Returns.drop(index=Returns.index[0], axis=0, inplace=True)

# modelling Ordinary Least Squares (OLS) regression
y = Returns['MMM_Returns']
x = Returns['DJI_Returns']
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
print(model.summary())

# plotting regression line
a = np.polyfit(Returns['DJI_Returns'], Returns['MMM_Returns'], 1)[0]
b = np.polyfit(Returns['DJI_Returns'], Returns['MMM_Returns'], 1)[1]
print('y = ' + '{:.4f}'.format(b) + ' + {:.4f}'.format(a) + 'x')
plt.scatter(Returns['DJI_Returns'], Returns['MMM_Returns'], color='pink')
plt.plot(Returns['DJI_Returns'], a*Returns['DJI_Returns']+b, color='purple')
plt.xlabel('DJI_Returns')
plt.ylabel('MMM_Returns')
plt.show()
