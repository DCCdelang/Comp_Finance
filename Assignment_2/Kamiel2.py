#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import *
from scipy.stats import norm
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
    ST = S* (np.exp( (r-0.5*sigma**2)*T + sigma*np.sqrt(T)*Z))
    
    return max(K-ST, 0)

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
# print(black_scholes_p(S,N,T,sigma,r,K))



# option_prices = []
# value = []
# value2 = []
# time = []

# n = [100,500, 1000,5000, 10000, 50000, 100000, 500000, 1000000]
# data = pd.read_csv(f'Monte_carlo.csv')
# sns.lineplot(data=data, x="Price", y="Values")
# plt.xscale("log") 
# # plt.show()
# data = pd.read_csv(f'Monte_carlo.csv')
# sns.lineplot(data=data, x="Price", y="Values")
# plt.xlabel("Paths", fontsize=14)
# plt.ylabel("Option price", fontsize=14)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.xscale("log")
# plt.tight_layout()
# plt.savefig("Convergence_option_price.pdf")
# plt.show()

# data = pd.read_csv(f'Monte_carlo.csv')
# values = data.loc[data['Price'] == 1000000]["Values"]
# m = np.mean(values)
# s = np.std(values)
# se = s/sqrt(100)

# ci = [m+(se*1.96), m-(se*1.96)]

# ci2 = [m+(s*1.96), m-(s*1.96)]
# print(f"SE = {se} sd = {s} ci1 = {ci} ci2 = {ci2}")
# value = []
# for i in range(1000000):
#     approxList = ST(K, S, r, sigma, T)
#     # print(approxList)
#     value.append(np.exp(-r * T)* approxList)
# # time.append(N)
# m = np.mean(value)
# sd = np.std(value)

# se = sd/np.sqrt(1000000)

# ci = [m+(se*1.96), m-(se*1.96)]
# print(f"CI3 = {ci}, sd = {sd}, se = {se} mean = {m}")
# option_prices.append(option_price(K, S, r, sigma, T, value))


# plt.plot(n, option_prices)
# data = {"Values":option_prices, "Price":time}
# df = pd.DataFrame(data) 
# df.to_csv(f"Monte_carlo.csv")
# sns.lineplot(data=df, x="Time", y="Values")
# plt.xscale("log") 
# plt.savefig("Convergence_option_price.pdf")
# plt.show()



"""
1.1 Perform numerical tests for varying values for the strike and the volatility parameter.
"""
# K = 99
# S = 100
# r = 0.06
# sigma = 0.2
# T = 1
# N = 50


# option_prices = []
# time = []
# std_error = []
# mean = []
# confidence_interval = []

# sigmas = np.linspace(0.1, 1, 10)
# n = 1000000


# for sigma in list(sigmas):
#     print(sigma)
#     value = []
#     value2 = []
#     for i in range(n):
#         approxList = ST(K, S, r, sigma, T)
#         value2.append(np.exp(-r * T)* approxList)
#         value.append(approxList)
#     time.append(sigma)
#     option_prices.append(option_price(K, S, r, sigma, T, value))   
#     m = np.mean(value2)  
    
#     se = np.std(value2)/np.sqrt(n)
  
#     confidence_interval.append([round(m-(se*1.96),3), round(m+(se*1.96),3)])
#     std_error.append(round(se,3))
#     mean.append(round(m,3))
# data = {"Values":option_prices, "Sigma":time, "Mean":mean, "Confidence interval":confidence_interval, "Standard error": std_error}
# df = pd.DataFrame(data) 
# df.to_csv(f"Monte_carlo_volatility.csv")

# print(option_prices)

# plt.plot(sigmas, option_prices)
# plt.show()
#%%
"""
1.1 Perform numerical tests for varying values for the strike and the strike parameter.
"""
# K = 99
# S = 100
# r = 0.06
# sigma = 0.2
# T = 1
# N = 50


# Ks = np.linspace(95, 105, 11)
# option_prices = []
# time = []

# std_error = []
# mean = []
# ste = []
# confidence_interval = []
# n = 1000000


# for K in list(Ks):
#     print(K)
#     value = []
#     value2 = []
#     for i in range(n):
#         approxList = ST(K, S, r, sigma, T)
#         value2.append(np.exp(-r * T)* approxList)
#         value.append(approxList)
#     time.append(K)
#     option_prices.append(option_price(K, S, r, sigma, T, value))     
#     m = np.mean(value2)   
#     se = np.std(value2)/np.sqrt(n)
#     confidence_interval.append([round(m-(se*1.96),3), round(m+(se*1.96),3)])
#     std_error.append(round(se,3))
#     mean.append(round(m,3))
# print(se)
# data = {"Values":option_prices, "Sigma":time, "Mean":mean, "Confidence interval":confidence_interval, "Standard error":std_error}
# df = pd.DataFrame(data) 
# df.to_csv(f"Monte_carlo_strike.csv")


"""
2.1 
"""

K = 99
S = 100
r = 0.06
sigma = 0.2
T = 1
N=50
epsilons = [0.0001]
d = ((np.log(S/K) + ((r-(sigma**2)/2)) * T )/(sigma*np.sqrt(T)))
d_s = -N_(-d)
print("\n")
print("Black-Scholes = ", d_s)

option_prices = []

option_prices2 = []
value = []
value2 = []
time = [100,10000,100000]

np.random.seed(42)
for epsilon in epsilons:
    print(epsilon)
    for i in time:
        for _ in range(i):
            approxList = ST(K, S, r, sigma, T)
            value.append(approxList)
        option_prices.append(option_price(K, S, r, sigma, T, value))

np.random.seed(42)
for epsilon in epsilons:
    print(epsilon)
    for i in time:
        for _ in range(i):
            approx_list2 = ST(K, S + epsilon, r, sigma, T)
            value2.append(approx_list2)
        option_prices2.append(option_price(K, S + epsilon, r, sigma, T, value2))


print(option_prices, option_prices2)

for i in range(3):
    delta = abs((option_prices2[i] - option_prices[i])/epsilon)
    print("Delta = ", delta)
