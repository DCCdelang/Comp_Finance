#%%
"""
Assignment 2 Computational Finance
Author: Dante de Lang
"""
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import norm
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
rs = RandomState(MT19937(SeedSequence(123456789)))
import time

#%%
"""
Part 1: Basic Option Valuation
"""

def eulerMethod(S,T,K,r,vol):
    Z = np.random.normal()
    S_T = S * np.exp((r-0.5*vol**2)*T + vol*(T**0.5)*Z)
    return S_T

def call_payoff(S_T, K):
    return max(S_T - K, 0.0)

def put_payoff(S_T, K):
    return max(K - S_T, 0.0)

def N_func(x):
    return norm.cdf(x)

def N_func_deriv(x):
    return norm.pdf(x)

def black_scholes(vol, S, T, K, r):
    d1 = (np.log(S/K) + (r+0.5*vol**2)*T)/(vol*T**0.5)
    d2 = d1 - (vol*T**0.5)
    return - S * N_func(-d1) + np.exp(-r * T) * K * N_func(-d2)

def monte_Carlo(S,T,K,r,vol,factor_list,simulations):
    averages = []
    for factor in factor_list:
        N_samples = 10**factor
        for _ in range(simulations):
            samples = []
            for _ in range(N_samples):
                S_T = eulerMethod(S,T,K,r,vol)
                samples.append(call_payoff(S_T, K))
            average = np.exp(-r*T)*np.mean(samples)
            # std = np.std(samples)
            averages.append([N_samples,average])
    return averages
    # df = pd.DataFrame(averages,columns=["N_samples","value"])
    # return df

vol = 0.2
K = 99
S = 100
r = 0.06
T = 1
#%%
# First test

factor_list = [1,2,3,4,5,6]
simulations = 2

MC_list = monte_Carlo(S,T,K,r,vol,factor_list,simulations)
df = pd.DataFrame(MC_list,columns=["N_samples","value"])
# df.to_csv("MC_normal.csv")

print(df)

#%%
df = pd.read_csv("MC_normal.csv")
print(df.head())
BS_value = black_scholes(vol, S, T, K, r)
print("Analytical value", BS_value)

sns.lineplot(data=df,x="N_samples",y="value", label="MC")
plt.hlines(BS_value,10**factor_list[0],10**factor_list[-1],label="BS")
plt.xscale("log")
plt.legend()
plt.show()
# plt.pause(5)
# plt.close()

#%%
""" For different strike prices """

factor_list = [5]
simulations = 5
K_list = [30,40,45,50,60,70,80,90,99]
payoff_K = []
for K in K_list:
    payoff_list = monte_Carlo(S,T,K,r,vol,factor_list,simulations)
    for sim in range(simulations):
        payoff_K.append([K,payoff_list[sim][1]])
print(pd.DataFrame(payoff_K))
df = pd.DataFrame(payoff_K,columns=["K","value"])
df.replace(np.nan,0)

sns.lineplot(data = df,x="K",y="value")
plt.xlabel("K-value")
plt.ylabel("Option value")
plt.show()

#%%
""" For different Volitalities"""
factor_list = [5]
simulations = 5
vol_list = [0.1,0.2,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,7,10]

payoff_vol = []
for vol in vol_list:
    payoff_list = monte_Carlo(S,T,K,r,vol,factor_list,simulations)
    for sim in range(simulations):
        payoff_vol.append([vol,payoff_list[sim][1]])
print(pd.DataFrame(payoff_vol))
df = pd.DataFrame(payoff_vol,columns=["vol","value"])

sns.lineplot(data = df,x="vol",y="value")
plt.xlabel("Volatility")
plt.ylabel("Option value")
plt.show()

#%%
"""
Part II: Estimation of Sensitivities in MC
"""

# 1) Bump and Revalue method

# Analytical delta for european put option (call - 1)
BS_Put = (N_func(((np.log(S/K) + (r+0.5*vol**2)*T)/(vol*T**0.5))))-1
print("Analytical", BS_Put)

epsilon = [0.000001,0.00001,0.0001,0.001,0.01,0.1]
factor_list = [4,5,6,7]
bump_data =  [ [] for _ in range(len(factor_list)) ]
print(bump_data)

# Bump and revalue method to calculate delta for ordinary european put option
for pos in range(len(factor_list)):
    N_samples = 10**factor_list[pos]
    bump_data[pos].append(N_samples)
    for eps in epsilon:
        samples = []
        np.random.seed(42)
        for _ in range(N_samples):
            S_T = eulerMethod(S+eps,T,K,r,vol)
            samples.append(put_payoff(S_T,K))
        average = np.exp(-r*T)*np.mean(samples)
        SE = np.std(samples)/N_samples
        # print("price1:", average)

        samples2 = []
        np.random.seed(42)
        for _ in range(N_samples):
            S_T2 = eulerMethod(S,T,K,r,vol)
            samples2.append(put_payoff(S_T2, K))
        average2 = np.exp(-r*T)*np.mean(samples2)
        SE2 = np.std(samples2)/N_samples
        # print("price2:",average2)

        Delta = ((average-average2)/eps)
        print("Delta:",Delta)
        print("Error:", (Delta-BS_Put)/BS_Put,"\n")
        bump_data[pos].append([Delta, (Delta-BS_Put)/BS_Put])

df = pd.DataFrame(bump_data,columns= ["Samples","0.000001+error","0.00001+error","0.0001+error","0.001+error","0.01+error","0.1+error"])
df.to_csv("Bump_revalue_put.csv")


#%%

# 2) Digital option

N_samples = 100

# Binary pay off functions
def binary_call_payoff(S_T, K):
    if S_T >= K:
        return 1.0
    else:
        return 0.0

def binary_put_payoff(S_T, K):
    if S_T < K:
        return 1.0
    else:
        return 0.0

# Black scholes analytical delta for binary put and call
binary_BS_Call = (np.exp(-r*T)*N_func_deriv((np.log(S/K) + (r-0.5*vol**2)*T)/(vol*T**0.5)))/(S*vol*T**0.5)
print(binary_BS_Call,"\n")

binary_BS_Put = -(np.exp(-r*T)*N_func_deriv((np.log(S/K) + (r-0.5*vol**2)*T)/(vol*T**0.5)))/(S*vol*T**0.5)
print(binary_BS_Put,"\n")

factor_list = [4,5,6]
binary_bump_data = [ [] for _ in range(len(factor_list)) ]

# Bump and revalue method for binary put delta approximation
epsilon = [0.001, 0.01, 0.1,]
for pos in range(len(factor_list)):
    N_samples = 10**factor_list[pos]
    binary_bump_data[pos].append(N_samples)
    for eps in epsilon:
        payoff_list = []
        np.random.seed(100)
        for i in range(N_samples):
            # S_T = (N_func(-((np.log((S+eps)/K) + (r-0.5*vol**2)*T)/(vol*T**0.5))))
            S_T = eulerMethod(S+eps,T,K,r,vol)
            payoff = binary_put_payoff(S_T, K) 
            payoff_list.append(payoff)
        price = np.exp(-r*T)*np.mean(payoff_list)
        print("Price:", price)

        payoff_list2 = []
        np.random.seed(42)
        for i in range(N_samples):
            # S_T2 = (N_func(-((np.log(S/K) + (r-0.5*vol**2)*T)/(vol*T**0.5))))
            S_T2 = eulerMethod(S,T,K,r,vol)
            payoff2 = binary_put_payoff(S_T2, K) 
            payoff_list2.append(payoff2)
        price2 = np.exp(-r*T)*np.mean(payoff_list2)
        print("Price2:", price2)
        Delta = ((price-price2)/eps)
        print("Delta:",Delta)
        error = []
        for i in range(N_samples):
            error.append(payoff_list2[i]-payoff_list[i])
        se = (np.std(error)/(np.sqrt(N_samples)/eps))
        print("Error:",se,"\n")
        binary_bump_data[pos].append([Delta, se])
df = pd.DataFrame(binary_bump_data,columns= ["Samples","0.001","0.01","0.1"])
df.to_csv("Bump_revalue_put_binary_dif_3.csv")

#%%
# Pathwise method 

N_samples = 10000

# Pathwise method for approximation regular european put delta
delta_list = []
np.random.seed(42)
for i in range(N_samples):
    S_T = eulerMethod(S,T,K,r,vol)
    binary_payoff = np.exp(K-S_T)/(1+np.exp(K-S_T))**2
    delta_list.append(np.exp(-r*T)*binary_payoff*S_T/S)
delta = np.mean(delta_list)
std = np.std(delta_list)/(N_samples**0.5)
print("Delta:", delta, std)
print("Error", (binary_BS_Put+delta)/binary_BS_Put)

#%%
# Likelihood ratio method

def eulerMethod_Z(S,T,K,r,vol):
    Z = np.random.normal()
    S_T = S * np.exp((r-0.5*(vol**2))*T + vol*(T**0.5)*Z)
    return S_T, Z

def binary_put_payoff(S_T, K):
    if S_T < K:
        return 1.0
    else:
        return 0.0

N_samples = 100000

# Likelihood ratio method for binary digital delta approximation for european put
delta_list = []
np.random.seed(42)
for i in range(N_samples):
    S_T, Z = eulerMethod_Z(S,T,K,r,vol)
    binary_payoff = binary_put_payoff(S_T, K)
    delta_list.append(np.exp(-r*T)*binary_payoff*(Z/(S*vol*T**0.5)))
delta = np.mean(delta_list)
print("Delta:", delta)

print((delta-binary_BS_Put)/binary_BS_Put)
