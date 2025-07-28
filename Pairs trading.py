import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

def find_optimal_trading_pairs(tickers):
    # This returns the pair with the highest correlation
    def correlated_pair(corr_values):
        if not corr_values:
            return None, None
        # Returns the index of the maximum value
        max_pair = corr_values.idxmax()
        max_value = corr_values.max()
        
        # Only interested in highly correlated pairs
        if max_value < 0.8:
            corr_values = corr_values.drop(index = max_pair)
            max_pair, max_value = correlated_pair(corr_values)
            
        return max_pair, max_value

    def stat_test(asset_1, asset_2):
        # Augmented Dickey Fuller test for stationarity of time series data
        # Null hypothesis is that the relationships are non-stationary (we want stationary relationships)
        hash_map = {f'{asset_1}': adfuller(asset_1)[1],
                    f'{asset_2}': adfuller(asset_2)[1],
                    'Spread': adfuller(asset_1 - asset_2)[1],
                    'Ratio': adfuller(asset_1 / asset_2)[1]}

        # p-value to determine whether to reject the null hypothesis, if it is less tha this value, we reject it in favour of 
        # the alternate hypothesis that the time series are stationary
        p_value = 0.05
        res = []
        
        for key, value in hash_map.items():
            if value <= p_value:
                res.append(key)
        
        return res

    # Function to test each correlated pair (highest first) to see if the time series' are stationary and useful for making 
    # predictions, returns the pair and the metric which should be used to generate signal (ratio, spread, price etc.)
    def trial_pair(corr_values):
        pair, _ = correlated_pair(corr_values)

        if not pair:
            raise ValueError('No correlated pairs were found')
            
        result = stat_test(data[pair[0]], data[pair[1]])

        if result:
            return result, pair[0], pair[1]   
        else:
            corr_values = corr_values.drop(index = pair)
            return trial_pair(corr_values)

    # Downloads the close price data of each stock and adds it to the dataframe
    data = pd.DataFrame()
    names = list()
    for i in tickers:
        data = pd.concat([data, yf.download(i, start='2025-01-01', end=date.today())['Close']], axis = 1)
        names.append(i)
    data.columns = names

    corr_matrix = data.corr()

    # This makes a boolean matrix where the upper triangle is true. We only need to look at the upper triangle as the matrix is 
    # symetric, we also ignore the diagonal as this is just 1.
    mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool) #k = 1 is the diagonal offset

    # pd.where() keeps all values where a condition is true, in this case the condition is given by the boolean matrix
    # .stack() flattens the columns on top of rows, so at each row index we see the columns of that row, the index becomes the pair
    # of stocks for that position. All elements of the matrix are stacked on top of each other in a series
    corr_values = corr_matrix.where(mask).stack()

    return trial_pair(corr_values)

def trading_signals():
    pass

if __name__ == 'main':
    tickers = ['AAPL', 'GOOG', 'AMD', 'SPY', 'NFLX', 'BA', 'NKE', 'FB', 'MSFT', 'BRK-B', 'GS', 'AMZN']
    find_optimal_trading_pairs(tickers)

'''
#data for the 2 assets
ticker_1 = 'AAPL' #'NEE'
ticker_2 = 'GE' #'OGE'

data_1 = yf.download(ticker_1, start='2022-01-01', end=date.today())
asset_1 = data_1['Close']
asset_1 = asset_1.to_numpy().flatten()

data_2 = yf.download(ticker_2, start='2022-01-01', end=date.today())
asset_2 = data_2['Close']
asset_2 = asset_2.to_numpy().flatten()

#calculate correlation
def correlation(x, y):
    return np.corrcoef(x, y)

#calculate spread
historic_spread = asset_1[:300] - asset_2[:300]
mean_spread = np.mean(historic_spread)
sdev_spread = np.std(historic_spread)

model = LinearRegression()
model.fit(asset_2.reshape(-1, 1), asset_1)

beta = model.coef_[0]
#r_squared = 0.47

current_spread = asset_1[-1] - (beta * asset_2[-1])

#calculate z-score
z = (current_spread - mean_spread) / sdev_spread

#calculate profit
i = 0
open_pos = False
negative_z = False
balance = 1000

while i < len(asset_1):
    spread = asset_1[i] - (beta * asset_2[i])
    z = (spread - mean_spread) / sdev_spread

    if open_pos == False and z > 2:
        if balance > asset_2[i]:
            short_pos = asset_1[i]
            long_pos = asset_2[i]
            open_pos = True
        else:
            print('Insufficient funds to perform transaction')
            break

    if open_pos == False and z < -2:
        if balance > asset_1[i]:
            short_pos = asset_2[i]
            long_pos = asset_1[i]
            open_pos = True
            negative_z = True
        else:
            print('Insufficient funds to perform transaction')
            break

    if open_pos == True and z > -2 and z < 2:
        open_pos = False
        if negative_z:
            short_profit = short_pos - asset_2[i] 
            long_profit = asset_1[i] - long_pos
        else:
            short_profit = short_pos - asset_1[i] 
            long_profit = asset_2[i] - long_pos

        balance += (short_profit + long_profit)
        print(balance)
    i += 1
    
'''

'''
import matplotlib.pyplot as plt

plt.plot(asset_1, 'r')
plt.plot(asset_2, 'b')


plt.show()
'''