import numpy as np
import Data_Downloader as dd
from scipy.stats import norm
from datetime import date, datetime, timedelta

ticker = 'AAPL'
country = 'uk'
duration = 30

def european_call(S, K, tau, sigma, r):
    d_1 = (np.log(S/K) + (r + (sigma**2/2)) * (tau))/(sigma * np.sqrt(tau))
    d_2 = (np.log(S/K) + (r - (sigma**2/2)) * (tau))/(sigma * np.sqrt(tau))
    return S*norm.cdf(d_1) - K*np.exp(-r*(tau))*norm.cdf(d_2)
  
def european_put(S, K, tau, sigma, r):
    d_1 = (np.log(S/K) + (r + (sigma**2/2)) * (tau))/(sigma * np.sqrt(tau))
    d_2 = (np.log(S/K) + (r - (sigma**2/2)) * (tau))/(sigma * np.sqrt(tau))
    return K*np.exp(-r*tau)*norm.cdf(-d_2) - S*norm.cdf(-d_1)

def us_option(K, sigma, tau, r, option_type):
    S_steps = 150
    t_steps = 150

    S_max = K*2

    dS = S_max / S_steps
    dt = tau / t_steps

    #sets up the grid indecies
    S = np.linspace(0, S_max, S_steps + 1)
    t = np.linspace(tau, 0, t_steps + 1)

    V = np.zeros((S_steps +1, t_steps + 1))

    t_reversed = t
    #boundary conditions
    if option_type == 'call':
        V[:, -1] = np.maximum(S - K, 0)
        V[0, :] = 0
        V[-1, :] = S_max - K * np.exp(-r * t)

    if option_type == 'put':
        V[:, -1] = np.maximum(K - S, 0)
        V[-1, :] = 0
        V[0, :] = K * np.exp(-r * t)

    #coefficients of the derivatives
    alpha = 0.25 * dt * (sigma**2 * (np.arange(1, S_steps) / dS)**2 - r * np.arange(1, S_steps) / dS)
    beta = -0.5 * dt * (sigma**2 * (np.arange(1, S_steps) / dS)**2 + r)
    gamma = 0.25 * dt * (sigma**2 * (np.arange(1, S_steps) / dS)**2 + r * np.arange(1, S_steps) / dS)

    #tridiagonal matrices of the coefficients above
    A = np.diag(1 - beta) + np.diag(-gamma[:-1], 1) + np.diag(-alpha[1:], -1)
    B = np.diag(1 + beta) + np.diag(gamma[:-1], 1) + np.diag(alpha[1:], -1)

    #steps back through each time step from the payoff boundary (a.k.a t=expiry)
    for n in range(t_steps, 0, -1):
        b = B @ V[1:S_steps, n]

        V_guess = V[1:S_steps, n - 1].copy()
        if option_type == 'call':
            payoff = np.maximum(S[1:S_steps] - K, 0)
        if option_type == 'put':
            payoff = np.maximum(K - S[1:S_steps], 0)

        #this sections uses PSOR method to ensure option is exercised if that is most profitable at each grid point
        tolerance = 1e-6  
        omega = 1.2  
        max_iteration = 1000

        for _ in range(max_iteration):
            V_old = V_guess.copy()

            for i in range(1, S_steps - 2):
                residual = (b[i]
                            - A[i, i - 1] * V_guess[i - 1]
                            - A[i, i] * V_guess[i]
                            - A[i, i + 1] * V_guess[i + 1])
                V_guess[i] = max(payoff[i], V_guess[i] + omega * residual / A[i, i])

            if np.max(np.abs(V_guess - V_old)) < tolerance:
                break
        V[1:S_steps, n - 1] = V_guess

    return S, t, V

def implied_vol(ticker, K, q, expiry, option_type):

        calls, puts = dd.option_data(ticker, expiry)

        S = dd.current_price(ticker)
        #r = dd.risk_free_rate('uk', 30)
        r = 0.05
        T = datetime.strptime(expiry, '%Y-%m-%d').date()
        today = date.today()
        t = np.busday_count(today, T + timedelta(days=1))
        
        max_iter = 200
        tol = 0.00001
        if option_type == 'call':
            current_price = calls[calls['strike'] == K]['lastPrice'].values[0]
        if option_type == 'put':
            current_price = puts[puts['strike'] == K]['lastPrice'].values[0]
        
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

def implied_vol2(ticker, country, expiry):
    #equation for volatility from https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=f93ec305fdc3960860548db0934e8615ee01f881
    def calculation(K, C, T):
        X = K*np.exp(-r*T)
        eta = X/S
        #Checks how close to the money an option is. Rejects outside of this range as no real solutions exist
        if eta > 0.85 and eta < 1.15:
            alpha = ((np.sqrt(2*np.pi)/(1+eta)))*((2*C/S) + eta - 1)

            z = np.cos(1/3*np.arccos((3*alpha)/np.sqrt(32)))
        
            sigma1 = (2*np.sqrt(2)/np.sqrt(T))*z - (1/np.sqrt(T))*np.sqrt((8*z**2)-((6*alpha)/(np.sqrt(2)*z))) # Li nearly at the money

            return sigma1
        
        else:
            return None

    #Data for the underlying asset
    S = dd.current_price(ticker)

    #Interest rate / risk free rate
    r = dd.risk_free_rate(country, 30)

    #Imports a dataframe containing all call options for the given stock and expiry
    calls, _ = dd.option_data(ticker, expiry)
    
    #Calculates IV for each strike and places that in a list
    T = float((datetime.strptime(expiry, '%Y-%m-%d').date() - date.today()).days/252)
    iv_list = list()
    for i in range(len(calls)):
        K = calls.iloc[i, 0]
        C = calls.iloc[i, 1]
        iv = calculation(K, C, T)
        iv_list.append(iv)
    
    calls['impliedVol'] = iv_list

    #Weights each IV by its open value and then sums them to find the IV for the stock
    interest_sum = 0
    for index, _ in calls.iterrows():
        if not np.isnan(calls.iloc[index, 3]):
            interest_sum += calls.iloc[index, 2]

    weighted_iv = 0
    for index, _ in calls.iterrows():
        value = calls.iloc[index, 3]
        if not np.isnan(value):
            weight = calls.iloc[index, 2]/interest_sum
            weighted_iv += (value*weight)
    
    return weighted_iv


