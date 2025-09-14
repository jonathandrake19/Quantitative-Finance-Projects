import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# Given a list of stock tickers, this returns the optimal pair for pairs trading
def optimalTradingPairs(tickers): 
    # This returns the pair with the highest correlation
    def correlatedPair(corr_values):
        if corr_values.empty:
            return None, None
        
        # Returns the index of the maximum value
        max_pair = corr_values.idxmax()
        max_value = corr_values.max()
        
        # Only interested in highly correlated pairs
        if max_value < 0.8:
            corr_values = corr_values.drop(index = max_pair)
            max_pair, max_value = correlatedPair(corr_values)
 
        return max_pair, max_value

    def statTest(asset_1, asset_2):
        # Augmented Dickey Fuller test for stationarity of time series data
        # Null hypothesis is that the relationships are non-stationary (we want stationary relationships)
        hash_map = {f'{asset_1}': adfuller(asset_1)[1],
                    f'{asset_2}': adfuller(asset_2)[1],
                    'spread': adfuller(asset_1 - asset_2)[1],
                    'ratio': adfuller(asset_1 / asset_2)[1]}

        # p-value to determine whether to reject the null hypothesis, if it is less than this value, we reject it in favour of 
        # the alternate hypothesis that the time series are stationary
        p_value = 0.05
        res = []
        
        # returns the metrics (spread, ratio etc.) of the pair which are stationary
        for key, value in hash_map.items():
            if value <= p_value:
                res.append(key)
        
        return res

    # Function to test each correlated pair (highest first) to see if the time series' are stationary and useful for making 
    # predictions, returns the pair and the metric which should be used to generate signal (ratio, spread, price etc.)
    # If the first pair is non-stationary the function will recursively check the next most correlated pairs
    def trialPair(corr_values):
        # Given the list of pairs and their correlations, this call to the function returns the pair with the highest
        # correlation. Each time we remove a pair (i.e. didn't satisfy stationarity condition) we call the function again 
        # to find the next most correlated pair 
        pair, _ = correlatedPair(corr_values)

        if not pair:
            raise ValueError('No correlated pairs were found')

        # perform the statistical test on the the current most correlated pair
        result = statTest(data[pair[0]], data[pair[1]])

        if result:
            return result, pair[0], pair[1]   
        
        else:
            # removes the pair which was just tried
            corr_values = corr_values.drop(index = pair)
            return trialPair(corr_values)

    # Downloads the close price data of each stock and adds it to the dataframe
    data = pd.DataFrame()
    names = list()
    for i in tickers:
        data = pd.concat([data, yf.download(i, start='2024-01-01', end=date.today())['Close']], axis = 1)
        names.append(i)

    data.columns = names

    # correlation matrix for all pairs of stocks
    corr_matrix = data.corr()

    # This makes a boolean matrix where the upper triangle is true. We only need to look at the upper triangle as the matrix is 
    # symetric, we also ignore the diagonal as this is just 1.
    mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool) # k = 1 is the diagonal offset

    # pd.where() keeps all values where a condition is true, in this case the condition is given by the boolean matrix
    # .stack() the index becomes the pair of stocks for that position. All elements of the matrix are stacked on 
    # top of each other in a series
    corr_values = corr_matrix.where(mask).stack()

    # This will go through each pair in order of highest correlation until it finds one which statisfies the stationarity condiition
    # it willl return the first pair to satisfy the condition (i.e. the most correlated pair to satisfy the condition)
    return trialPair(corr_values)

# Given a pair of stocks this generates buy and sell signals based on a mean reverting trading stratergy
def tradingSignals(ticker1, ticker2, indicator):
    s1 = (yf.download(ticker1, start='2024-01-01', end=date.today())['Close']).to_numpy().flatten()
    s2 = (yf.download(ticker2, start='2024-01-01', end=date.today())['Close']).to_numpy().flatten()

    def movingAverage(index):
        if indicator == 'ratio':
            historicIndicator = s1[index - 252: index] / s2[index - 252: index]
        
        elif indicator == 'spread':
            historicIndicator = s1[index - 1: index] - s2[index - 100: index]

        mu = historicIndicator.mean()
        sigma = historicIndicator.std()

        return mu, sigma
    
    k = 1.75

    #plt.plot(s1)
    #plt.plot(s2)
    plt.plot((s1 / s2))
    
    # openPos keeps track of whether we have active trades
    openPos = False

    # keeps track of whether the trades were initiated by a negative z score
    negativeZ = False

    balance = 0
    i = 252
    mu, sigma = movingAverage(i)
    plt.axhline(mu)
    plt.show()
    while i < len(s1):
        # when the z score of the indicator exceeds a given multiple of the historical standard deviation, the deviation of the 
        # indicator is deemed to be significant and not just due to statistical noise
        x = s1[i] / s2[i]
        z = (x - mu) / sigma
        
        # if the z score of the indicator is greater than the positive threshold we short stock 1 and take a long position in stock 2.
        # In this case either the price of stock 1 is much higher than expected (shorting makes money), or the price of stock 2 
        # is much lower than expected (long position makes money)
        if openPos == False and z >= k * sigma:
            shortPos = s1[i]
            longPos = s2[i]
            openPos = True

        # if the z score is less than the negative threshold we go long in stock 1 and short stock 2. This is the opposite situation
        # to the previous case
        elif openPos == False and z <= -k * sigma:
            shortPos = s2[i]
            longPos = s1[i]
            negativeZ = True
            openPos = True
        
        #elif openPos == False:
            #mu, sigma = movingAverage(i)
        
        elif openPos == True and -k * sigma < z < k * sigma:
            if negativeZ:
                shortProfit = shortPos - s2[i]
                longProfit = s1[i] - longPos
            
            else:
                shortProfit = shortPos - s1[i]
                longProfit = s2[i] - longPos
            
            openPos = False
            negativeZ = False

            #mu, sigma = movingAverage(i)
            balance += shortProfit + longProfit
            print(balance)
        mu, sigma = movingAverage(i)
        i += 1

    #print(balance)


#if __name__ == 'main':
tickers = ['AAPL', 'GOOG', 'AMD', 'SPY', 'NFLX', 'BA', 'NKE', 'FB', 'MSFT', 'BRK-B', 'GS', 'AMZN']

indicator, s1, s2 = optimalTradingPairs(tickers)
tradingSignals(s1, s2, indicator[0])

