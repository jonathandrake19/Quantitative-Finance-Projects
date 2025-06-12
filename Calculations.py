import numpy as np
import Data_Downloader as dd
from scipy.stats import norm
from datetime import date, datetime, timedelta
import pandas as pd

def european_call(S, K, tau, sigma, r):
    d_1 = (np.log(S/K) + (r + (sigma**2/2)) * (tau))/(sigma * np.sqrt(tau))
    d_2 = (np.log(S/K) + (r - (sigma**2/2)) * (tau))/(sigma * np.sqrt(tau))
    return S*norm.cdf(d_1) - K*np.exp(-r*(tau))*norm.cdf(d_2)
  
def european_put(S, K, tau, sigma, r):
    d_1 = (np.log(S/K) + (r + (sigma**2/2)) * (tau))/(sigma * np.sqrt(tau))
    d_2 = (np.log(S/K) + (r - (sigma**2/2)) * (tau))/(sigma * np.sqrt(tau))
    return K*np.exp(-r*tau)*norm.cdf(-d_2) - S*norm.cdf(-d_1)

def implied_vol(ticker, K, q, expiry, option_type):

        calls, puts = dd.option_data(ticker, expiry)

        S = dd.current_price(ticker)
        r = dd.risk_free_rate('uk', 30)
        T = datetime.strptime(expiry, '%Y-%m-%d').date()
        today = date.today()
        t = np.busday_count(today, T + timedelta(days=1))

        max_iter = 200
        tol = 0.00001
        if option_type == 'call':
            current_price = calls[calls['strike'] == K]['ask'].values[0]
        if option_type == 'put':
            current_price = puts[puts['strike'] == K]['ask'].values[0]
        
        vol_old = 0.3

        for _ in range(max_iter):
            d_1 = (np.log(S/K) + (r + (vol_old**2/2)) * (t))/(vol_old * np.sqrt(t))
            vega = S * np.exp(-q * t) * np.sqrt(t) * norm.pdf(d_1)

            if option_type == 'call':
                bs_price = european_call(S, K, t, vol_old/np.sqrt(252), r/252)
            if option_type == 'put':
                bs_price = european_put(S, K, t, vol_old/np.sqrt(252), r/252)

            vol_new = vol_old - ((bs_price - current_price) / vega)

            if vol_new <= 0:
                return vol_old

            if abs(vol_old - vol_new) < tol:
                break

            vol_old = vol_new

        return vol_new

def implied_vol_surface(ticker, q, expiry, today, option_type):

        calls, puts = dd.option_data(ticker, expiry)

        S = dd.current_price(ticker)
        r = 0.05
        T = datetime.strptime(expiry, '%Y-%m-%d').date()
        t = np.busday_count(today, T + timedelta(days=1)) / 252

        max_iter = 200
        tol = 0.00001
        if option_type == 'call':
            data = calls
        if option_type == 'put':
            data = puts
        
        K_list = []
        Vol_list = []

        for row in data.itertuples(index = True):
            K = row[1]
            if K/S < 0.8 or K/S > 1.2:
                continue
                
            if row[4] != 0 and row[3] != 0:
                current_price = 0.5 * (row[4] + row[3])

            else:
                current_price = row[2]

            if current_price <= 0 or np.isnan(current_price):
                continue

            vol_old = np.clip(np.sqrt(2 * abs(np.log(S / K) + r * t) / t), 0.01, 2.0)
            for _ in range(max_iter):
                d_1 = (np.log(S/K) + (r + (vol_old**2/2)) * (t))/(vol_old * np.sqrt(t))
                vega = S * np.exp(-q * t) * np.sqrt(t) * norm.pdf(d_1)

                if option_type == 'call':
                    bs_price = european_call(S, K, t, vol_old, r)
                if option_type == 'put':
                    bs_price = european_put(S, K, t, vol_old, r)

                if vega < 0.01:
                    break

                vol_new = vol_old - ((bs_price - current_price) / vega)
                
                if vol_new <= 0:
                    vol_new = vol_old
                    break

                if abs(vol_old - vol_new) < tol:
                    break

                vol_old = vol_new
            
            K_list.append(K)
            Vol_list.append(vol_new)

        K_array = np.array(K_list)
        Vol_array = np.array(Vol_list)
        return K_array, Vol_array
