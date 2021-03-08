#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import *
from scipy.stats import norm, gmean
import seaborn as sns

K = 99
S = 100
r = 0.06
sigma = 0.2
T = 1
N = 50
#%%

def option_price(K, S, r, sigma, T, pay_offs):
    return np.exp(-r * T)*(np.mean(pay_offs))

def ST(K, S, r, sigma, T):
   
    Z = np.random.normal()
    ST = S* (np.exp( (r-0.5*(sigma**2))*T + sigma*np.sqrt(T)*Z))
    
    return max(K - ST, 0)

def digital(K, S, r, sigma, T):
   
    Z = np.random.normal()
    ST = S* (np.exp( (r-0.5*(sigma**2))*T + sigma*np.sqrt(T)*Z))
    
    if max(K - ST, 0) > 0 :
        return 1, ST
    else:
        return 0, ST

def N_(x):
    return norm.cdf(x)

def black_scholes_c(S,N,T,sigma,r,K):
    d = ((np.log(S/K) + ((r-(sigma**2)/2)) * T )/(sigma*np.sqrt(T)))
    d_1 = d + sigma* np.sqrt(T)
    return (S*N_(d_1)) - (K*np.exp(-r*T)*N_(d))

def black_scholes_p(S,N,T,sigma,r,K):
    d = ((np.log(S/K) + ((r-(sigma**2)/2)) * T )/(sigma*np.sqrt(T)))
    d_1 = d + sigma* np.sqrt(T)
    return  (K*np.exp(-r*T)*N_(-d)) - (S*N_(-d_1))


"""
1.1 Carry out convergence studies by increasingthe number of trials.  How do your results compare with the results obtained in assignment 1?
"""
# Print analytical value of black sholes
print(black_scholes_p(S,N,T,sigma,r,K))

# Initialize amount of paths
n = [100,500, 1000,5000, 10000, 50000, 100000, 500000, 1000000]
a = [(black_scholes_p(S,N,T,sigma,r,K))]*len(n)
data_bs = {"Values":a, "Price":n}
values = []
time = []

# Do Monte Carlo method
for i in range(100):
    for path in n:
        value = []
        for i in range(path):
            approxList = ST(K, S, r, sigma, T)
            value.append(approxList)
        time.append(path)
        values.append(option_price(K, S, r, sigma, T, value))

data = {"Values":values, "Price":time}
df = pd.DataFrame(data) 
df.to_csv(f"Monte_carlo.csv")


data = pd.read_csv(f'Monte_carlo.csv')


sns.lineplot(data=data, x="Price", y="Values", label ="Monte Carlo")
sns.lineplot(data=data_bs, x="Price", y="Values", label ="Analytical")
plt.xlabel("Paths", fontsize=14)
plt.ylabel("Option price", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xscale("log")
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig("Convergence_option_price.pdf")
plt.show()

data = pd.read_csv(f'Monte_carlo.csv')
values = data.loc[data['Price'] == 1000000]["Values"]
m = np.mean(values)
s = np.std(values)
se = s/sqrt(100)

ci = [m+(se*1.96), m-(se*1.96)]

ci2 = [m+(s*1.96), m-(s*1.96)]
print(f"SE = {se} sd = {s} ci1 = {ci} ci2 = {ci2}")



"""
1.1 Perform numerical tests for varying values for the strike and the volatility parameter.
"""
K = 99
S = 100
r = 0.06
sigma = 0.2
T = 1
N = 50


option_prices = []
time = []
std_error = []
mean = []
confidence_interval = []

sigmas = np.linspace(0.1, 1, 10)
n = 1000000


for sigma in list(sigmas):
    print(sigma)
    value = []
    value2 = []
    for i in range(n):
        approxList = ST(K, S, r, sigma, T)
        value2.append(np.exp(-r * T)* approxList)
        value.append(approxList)
    time.append(sigma)
    option_prices.append(option_price(K, S, r, sigma, T, value))   
    m = np.mean(value2)  
    
    se = np.std(value2)/np.sqrt(n)
  
    confidence_interval.append([round(m-(se*1.96),3), round(m+(se*1.96),3)])
    std_error.append(round(se,3))
    mean.append(round(m,3))
data = {"Values":option_prices, "Sigma":time, "Mean":mean, "Confidence interval":confidence_interval, "Standard error": std_error}

df.to_csv(f"Monte_carlo_volatility.csv")

print(option_prices)

plt.plot(sigmas, option_prices)
plt.show()
#%%
"""
1.1 Perform numerical tests for varying values for the strike and the strike parameter.
"""
K = 99
S = 100
r = 0.06
sigma = 0.2
T = 1
N = 50


Ks = np.linspace(95, 105, 11)
option_prices = []
time = []

std_error = []
mean = []
ste = []
confidence_interval = []
n = 1000000


for K in list(Ks):
    print(K)
    value = []
    value2 = []
    for i in range(n):
        approxList = ST(K, S, r, sigma, T)
        value2.append(np.exp(-r * T)* approxList)
        value.append(approxList)
    time.append(K)
    option_prices.append(option_price(K, S, r, sigma, T, value))     
    m = np.mean(value2)   
    se = np.std(value2)/np.sqrt(n)
    confidence_interval.append([round(m-(se*1.96),3), round(m+(se*1.96),3)])
    std_error.append(round(se,3))
    mean.append(round(m,3))
print(se)
data = {"Values":option_prices, "Sigma":time, "Mean":mean, "Confidence interval":confidence_interval, "Standard error":std_error}
df = pd.DataFrame(data) 
df.to_csv(f"Monte_carlo_strike.csv")
