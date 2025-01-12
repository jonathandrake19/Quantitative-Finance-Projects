import matplotlib.pyplot as plt
import numpy as np
import Data_Downloader as dd
import time
#General info on investment e.g. company, timeframe, location etc.
ticker = 'AAPL'
country = 'uk'
duration = '1Y'
#General info on investment e.g. company, timeframe, location etc.
ticker = 'AAPL'
country = 'uk'
duration = '1Y'
if __name__ == "__main__":
    start_time = time.time()
#Info on time series for the Monte Carlo Simulation
    T = 1/12
    N = 21
    dt = T/N
    t = np.linspace(0, T, N+1)
    num_simulations = 10000

    #Parameters for SDE for stock price
    s_0 = dd.current_price(ticker)
    #mu =  dd.risk_free_rate(country, duration) 
    #s_0 = 250
    mu = 0.05
    sigma = 0.2743508826432856/np.sqrt(12)

    nu_dt = (mu - (sigma**2/2))*dt
    sigma_dt = sigma*np.sqrt(dt)

    #Monte Carlo simulation
    z = np.random.normal(size=(N, num_simulations))
    delta_s = np.exp(nu_dt + sigma_dt*z)
    ln_st = np.log(s_0) + np.cumsum(delta_s, axis=0)
    ln_st = np.concatenate((np.full(shape=(1,num_simulations), fill_value=np.log(s_0)), ln_st))
    #print(ln_st)
  
    end_time = time.time()
    print(f"Script runtime: {end_time - start_time:.4f} seconds")



'''
    #Graphing results
    price = np.array(prices)
    #for i in prices: 
        #plt.plot(t, i)
    #plt.show()
    final_prices = []
    for i in price:
        final = i[-1]
        final_prices.append(final)
    plt.hist(final_prices, bins=100)
    plt.show()
'''