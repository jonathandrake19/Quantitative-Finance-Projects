import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import Data_Downloader as dd
from scipy.stats import norm, poisson
from scipy.optimize import minimize

ticker = 'AAPL'
# calculates the parameters for the poisson process given how volatile the underlying stock is
def poisson_intensity(ticker):
    tk = yf.Ticker(ticker)
    price = tk.history(period = '3y')['Close']

    log_returns = np.log(price / price.shift(1))
    log_returns = log_returns.dropna()

    volatility = np.std(log_returns)
    k = 3

    # if the price changes by more than k standard deviations from 1 day to another, this is considered a jump
    jump_count = 0
    jumps = []
    for value in log_returns:
        if abs(value) >= k * volatility:
            jump_count += 1
            jumps.append(value)

    mean_jump = np.mean(jumps)
    jump_stdev = np.std(jumps)

    # lambda is the rate jumps occur at
    lambd = jump_count / len(log_returns)
    return lambd, mean_jump, jump_stdev

def likelihood_function(params, log_returns, dt):
    mu, sigma, lam, mu_y, sigma_y = params
    # cap the number of jumps as 5 to avoid infinities. Reasonable assumption as multiple jumps in an interval dt is small
    k = 5

    likelihoods = np.zeros(len(log_returns))

    for i in range(k + 1):
        # probability of there being k jumps
        prob_k = poisson.pmf(i, lam * dt)

        # if there are k jumps the mean drift will shift by k*mu_y, same with variance
        mean_k = mu * dt - 0.5 * sigma**2 * dt + k * mu_y
        var_k = sigma**2 * dt + k * sigma_y**2
        std_k = np.sqrt(var_k)

        # likelihood is the sum over all k of P(K=k) * N(return). Vectorised operation calulated for each log return
        likelihoods += prob_k * norm.pdf(log_returns, mean_k, std_k)

    # negative as optimizer in SciPy minimizes rather than maximizes. Minimizing negative log-likelihood is equivalent to 
    # maximizing log-likelihood
    log_likelihood = -np.sum(np.log(likelihoods + 1e-10))
    return log_likelihood

tk = yf.Ticker(ticker)
price = tk.history(period = '3y')['Close']

# log returns for the given time period before the beginning of the model
price = price.iloc[:-100]
log_returns = np.log(price / price.shift(1))

# parameters for the SDE / Brownian motion of the stock price
#S_0 = dd.current_price(ticker)
S_0 = price.iloc[-100]
mu = 0.05/252
sigma = 0.25/np.sqrt(252)
T = 100
dt = 1

num_simulations = 100

# parameters of the jumps (given by a Poisson process)
intensity, mu_y, sigma_y = poisson_intensity(ticker)

initial_params = [mu, sigma, intensity, mu_y, sigma_y]
# constraints are that standard deviation and lambda for Poisson distributions must be > 0
constraints = (
    {'type': 'ineq', 'fun': lambda x: x[1]},  
    {'type': 'ineq', 'fun': lambda x: x[2]},  
    {'type': 'ineq', 'fun': lambda x: x[4]}   
)

# Maximum Likelihood Estimation (MLE) to find the best parameters
result = minimize(likelihood_function, initial_params, args=(log_returns, dt), 
                  method='L-BFGS-B', constraints=constraints)

optimal_params = result.x
mu_opt, sigma_opt, lambda_opt, mu_y_opt, sigma_y_opt = optimal_params

# matrix of random standard normally distributed numbers for the Weiner process
z = np.random.normal(size = (T, num_simulations))

# matrix of random values representing when jumps occur
N_t = np.random.poisson(lam= lambda_opt, size = (T, num_simulations))

# initialise the matrix which stores the log of the jump size at each time step
log_jumps = np.zeros((T, num_simulations))

# iterate through each time position and simulation and calculate the jump size at every position indicated by the N_t matrix
for t in range(T):
    for sim in range(num_simulations):
        if N_t[t, sim] > 0:
            jump_sizes = np.random.lognormal(mu_y_opt, sigma_y_opt)
            log_jumps[t, sim] = np.log(jump_sizes)

# delta_s is the change in price at each time step for each simulation. It is the value of dS in the SDE
# delta_s has a T x num_simulations shape as this is the shape of z and log_jumps
delta_s = ((mu_opt - 0.5 * sigma_opt**2) * dt) + (sigma_opt * np.sqrt(dt) * z) + log_jumps

# each row in delta_s is an entire simulation so summing over each row gives the total log change in price at each time point
ln_st = np.log(S_0) + np.cumsum(delta_s, axis = 0)

# add the log of the starting price to the beginning of all rows
ln_st = np.concatenate((np.full(shape = (1, num_simulations), fill_value = np.log(S_0)), ln_st))

# create the price matrix by raising e to each value in ln_st
S_t = np.exp(ln_st)

# average over the price values for all simulations at time t = T 
final_values = np.mean(S_t[-1,:])

print(final_values)
# only plot the first ~100 simulations or otherwise the figure is unreadable
plt.figure(figsize=(10, 6))
for i in range(100):  
    plt.plot(S_t[:, i], label=f"Sim {i+1}")

plt.xlabel('Time (days)')
plt.ylabel('Price ($)')
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
