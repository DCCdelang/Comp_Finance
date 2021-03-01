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
    # np.random.seed(42)
    Z = np.random.normal()
    ST = S* (np.exp( (r-0.5*sigma**2)*T + sigma*np.sqrt(T)*Z))
    
    return max(K-ST, 0)

def N_(x):
    #'Cumulative distribution function for the standard normal distribution'

    # print((1.0 + erf(x / sqrt(2.0))) / 2.0)
    
    return norm.cdf(x)

def black_scholes_c(S,N,T,sigma,r,K):
    d = ((np.log(S/K) + ((r-(sigma**2)/2)) * T )/(sigma*np.sqrt(T)))
    d_1 = d + sigma* np.sqrt(T)
    return (S*N_(d_1)) - (K*np.exp(-r*T)*N_(d))

def black_scholes_p(S,N,T,sigma,r,K):
    d = ((np.log(S/K) + ((r-(sigma**2)/2)) * T )/(sigma*np.sqrt(T)))
    d_1 = d + sigma* np.sqrt(T)
    return  (K*np.exp(-r*T)*N_(-d)) - (S*N_(-d_1))

def asian_anal(S,N,T,sigma,r,K):
    sigma_tilde = sigma * sqrt((2*N+1)/(6*(N+1)))
    r_tilde = ((r-sigma/2)+sigma_tilde)/2

    d_tilde_1 = ((np.log(S/K) + ((r_tilde+(sigma_tilde**2)/2)) * T )/(sigma_tilde*np.sqrt(T)))
    d_tilde_2 = ((np.log(S/K) + ((r_tilde-(sigma_tilde**2)/2)) * T )/(sigma_tilde*np.sqrt(T)))

    return np.exp(-r*T)*(S*np.exp(r_tilde*T)*N_(d_tilde_1)-K*N_(d_tilde_2))

    
def asian_MC(S,N,T,r,K, n, type = "geometric"):
    summa = 0
    for i in range(n):
        S_ti_ar = 0
        S_ti_geo = 1
        if type == "arithmetic":
            for j in range(N):
                Z = np.random.normal()
                T_i = j*T/N
                ST = S * (np.exp( (r-0.5*sigma**2)*T_i + sigma*np.sqrt(T_i)*Z))
                S_ti_ar = S_ti_ar + ST
            summa = summa + max(0, S_ti_ar/(N+1)-K)
        elif type == "geometric":
            for j in range(N):
                Z = np.random.normal()
                T_i = j*T/N
                ST = S * (np.exp( (r-0.5*sigma**2)*T_i + sigma*np.sqrt(T_i)*Z))
                S_ti_geo = S_ti_geo * ST
                #print(S_ti_geo)
            summa = summa + max(0, S_ti_geo**(1/(N+1))-K)
            #print(summa)
    return np.exp(-r * T) * summa/n

#%%
#### ASIAN OPTION
K = 99
S = 100
r = 0.06
sigma = 0.2
T = 1
N = 50
n = 10000

asian_anal = asian_anal(S,N,T,sigma,r,K)
print(asian_anal)
asian_geom = asian_MC(S,N,T,r,K, n, type = "geometric")
print(asian_geom)

#%%
n = [100, 500, 1000,5000, 10000, 50000, 100000, 500000, 1000000]
asian_MC_list = []
for n in range(n):
    asian_MC = asian_arith(S,N,T,r,K,n)
    asian_MC_list = asian_MC_list.append(asian_MC)
    asian_geom_list = asian_geom(S,N,T,sigma,r,K)

plt.plot(n, asian_MC_list, label = "Arithmetic")
plt.plot(n, asian_geom_list, label = "Geometric")
plt.legend()
plt.show()
#%%
asian_arith_MC = asian_arith(S,N,T,r,K, m, n)
asian_anal = asian_anal(S,N,T,sigma,r,K)
asian_cv = asian_arith_MC + asian_anal + asian_geom_MC




#%%
"""
1.1 Carry out convergence studies by increasingthe number of trials.  
How do your results compare with the results obtained in assignment 1?
"""
print(black_scholes_p(S,N,T,sigma,r,K))

K = 99
S = 100
r = 0.06
sigma = 0.2
T = 1

option_prices = []
value = []
time = []
n = [100,500, 1000,5000, 10000, 50000, 100000, 500000, 1000000]

for i in range(5):
    for N in list(n):
        for i in range(int(N)):
            approxList = ST(K, S, r, sigma, T)
            # print(approxList)
            value.append(approxList)
        time.append(N)
        option_prices.append(option_price(K, S, r, sigma, T, value))


# plt.plot(n, option_prices)
data = {"Values":option_prices, "Time":time}
df = pd.DataFrame(data) 
df.to_csv(f"Monte_carlo.csv")
sns.lineplot(data=df, x="Time", y="Values")
plt.xscale("log") 
plt.savefig("Convergence_option_price.pdf")
plt.show()

"""
1.1 Perform numerical tests for varying values for the strike and the volatility parameter.
"""


option_prices = []
value = []
time = []
sigmas = np.linspace(0.01, 10, 10)

for sigma in list(sigmas):
    value = []
    for i in range(1000000):
        approxList = ST(K, S, r, sigma, T)
        # print(approxList)
        value.append(approxList)
    time.append(sigma)
    option_prices.append(option_price(K, S, r, sigma, T, value))
    print(option_prices)


print(option_prices)

plt.plot(sigmas, option_prices)
plt.show()
#%%
option_prices = []

time = []
Ks = np.linspace(80, 120, 10)

for K in list(Ks):
    value = []
    for i in range(1000000):
        approxList = ST(K, S, r, sigma, T)
        # print(approxList)
        value.append(approxList)
    time.append(sigma)
    option_prices.append(option_price(K, S, r, sigma, T, value))

plt.plot(Ks, option_prices)
plt.show()

#%%
K = 99
S = 100
r = 0.06
sigma = 0.2
T = 1
N=50
epsilons = [0.01,0.02,0.5]

option_prices = []

option_prices2 = []
value = []
value2 = []
time = []

for epsilon in epsilons:
    for i in range(1000000):
        approxList = ST(K, S, r, sigma, T)
        approx_list2 = ST(K, S+epsilon, r, sigma, T)
        value.append(approxList)
        value2.append(approx_list2)
    option_prices.append(option_price(K, S, r, sigma, T, value))
    option_prices2.append(option_price(K, S, r, sigma, T, value2))

print(option_prices)

for i in range(3):
    delta = (option_prices2[i] - option_prices[i])/epsilon
    print(delta)
# %%
