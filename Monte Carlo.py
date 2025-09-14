import matplotlib.pyplot as plt
import numpy as np
import Data_Downloader as dd
import time

ticker = 'AAPL'

if __name__ == "__main__":
    start_time = time.time()

    # info on time series for the Monte Carlo Simulation
    # number of time periods in 1 year (12 months, 365 days etc.)
    periods = 12
    T = 1 / periods

    # trading days in period
    N = 21

    # time interval
    dt = T/N
    t = np.linspace(0, T, N+1)
    
    num_simulations = 10000

    # parameters for SDE for stock price
    s_0 = dd.current_price(ticker)
    mu = 0.05
    sigma = 0.2/np.sqrt(12)

    nu_dt = (mu - (sigma**2/2))*dt
    sigma_dt = sigma*np.sqrt(dt)

    # Monte Carlo simulation
    z = np.random.normal(size=(N, num_simulations))
    delta_s = np.exp(nu_dt + sigma_dt*z)
    ln_st = np.log(s_0) + np.cumsum(delta_s, axis=0)
    ln_st = np.concatenate((np.full(shape=(1,num_simulations), fill_value=np.log(s_0)), ln_st))

    # final prices
    s_t = np.exp(ln_st
                 
    end_time = time.time()
    print(f"Script runtime: {end_time - start_time:.4f} seconds")

'''
    # graphing results as a time series
    plt.figure(figsize=(10, 6))
    # only plot a ~100 price paths or graph become unreadable
    for i in range(100):  
        plt.plot(S_t[:, i], label=f"Sim {i+1}")
    
    plt.xlabel('Time')
    plt.ylabel('Price ($)')
    plt.show()

'''
