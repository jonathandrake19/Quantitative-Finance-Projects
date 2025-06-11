import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import Data_Downloader as dd
from scipy.stats import norm, poisson
from scipy.optimize import minimize

ticker = 'AAPL'
def poisson_intensity(ticker):
    tk = yf.Ticker(ticker)
    price = tk.history(period = '3y')['Close']

    log_returns = np.log(price / price.shift(1))
    log_returns = log_returns.dropna()

    volatility = np.std(log_returns)
    k = 3

    jump_count = 0
    jumps = []
    for value in log_returns:
        if abs(value) >= k * volatility:
            jump_count += 1
            jumps.append(value)

    mean_jump = np.mean(jumps)
    jump_stdev = np.std(jumps)

    lambd = jump_count / len(log_returns)
    return lambd, mean_jump, jump_stdev

def likelihood_function(params, log_returns, dt):
    mu, sigma, lam, mu_y, sigma_y = params
    k = 5

    likelihoods = np.zeros(len(log_returns))

    for i in range(k + 1):
        prob_k = poisson.pmf(i, lam * dt)
        mean_k = mu * dt - 0.5 * sigma**2 * dt + k * mu_y
        var_k = sigma**2 * dt + k * sigma_y**2
        std_k = np.sqrt(var_k)

        likelihoods += prob_k * norm.pdf(log_returns, mean_k, std_k)

    log_likelihood = -np.sum(np.log(likelihoods + 1e-10))
    return log_likelihood

lambda_dict = {}
mu_y_dict = {}
sigma_y_dict = {}

if ticker not in lambda_dict.keys():
    func_values = poisson_intensity(ticker)
    lambda_dict[ticker] = func_values[0]
    mu_y_dict[ticker] = func_values[1]
    sigma_y_dict[ticker] = func_values[2]

#Parameters for Merton Jump model
tk = yf.Ticker(ticker)
price = tk.history(period = '3y')['Close']
price = price.iloc[:-100]
log_returns = np.log(price / price.shift(1))

#S_0 = dd.current_price(ticker)
S_0 = price.iloc[-100]
mu = 0.05/252
sigma = 0.25/np.sqrt(252)
T = 100
dt = 1
Num_simulations = 1000

intensity = lambda_dict[ticker]
mu_y = mu_y_dict[ticker]
sigma_y = sigma_y_dict[ticker]



initial_params = [mu, sigma, intensity, mu_y, sigma_y]
constraints = (
    {'type': 'ineq', 'fun': lambda x: x[1]},  
    {'type': 'ineq', 'fun': lambda x: x[2]},  
    {'type': 'ineq', 'fun': lambda x: x[4]}   
)

result = minimize(likelihood_function, initial_params, args=(log_returns, dt), 
                  method='L-BFGS-B', constraints=constraints)

optimal_params = result.x
mu_opt, sigma_opt, lambda_opt, mu_y_opt, sigma_y_opt = optimal_params

z = np.random.normal(size = (T, Num_simulations))

N_t = np.random.poisson(lam= lambda_opt, size = (T, Num_simulations))

log_jumps = np.zeros((T, Num_simulations))

for t in range(T):
    for sim in range(Num_simulations):
        if N_t[t, sim] > 0:
            jump_sizes = np.random.lognormal(mu_y_opt, sigma_y_opt, N_t[t, sim])
            log_jumps[t, sim] = np.sum(np.log(jump_sizes))

delta_s = ((mu_opt - 0.5 * sigma_opt**2) * dt) + (sigma_opt * np.sqrt(dt) * z) + log_jumps
ln_st = np.log(S_0) + np.cumsum(delta_s, axis = 0)
ln_st = np.concatenate((np.full(shape = (1, Num_simulations), fill_value = np.log(S_0)), ln_st))

S_t = np.exp(ln_st)

final_values = np.mean(S_t[-1,:])

print(final_values)
plt.figure(figsize=(10, 6))
for i in range(100):  
    plt.plot(S_t[:, i], label=f"Sim {i+1}")
plt.show()


#for if you want to convert to BSE style. NEEDS WORK ISN'T FULLY CORRECT
'''
k = np.exp(mu_y + 0.5 * sigma_y**2) - 1 
S_sum = 0
for n in range(2):
    z = np.random.normal(size = (1, T))
    S_n = S_0 * np.exp(-intensity*k + n*mu_y + ((n*sigma_y**2)/2))
    sigma_n = np.sqrt(sigma**2 + (n*sigma_y**2/dt))

    delta_s = ((mu - 0.5 * sigma_n**2) * dt) + (sigma_n * dt) * z
    ln_st = np.log(S_n) + np.cumsum(delta_s)

    p_poisson = (intensity**n * np.exp(-intensity)) / factorial(n)
    S_sum += (np.exp(ln_st[-1]) * p_poisson)
'''
