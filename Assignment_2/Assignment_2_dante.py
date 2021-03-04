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

def eulerMethodPut(S,T,K,r,vol):
    Z = np.random.normal()
    S_T = S * np.exp((r-0.5*vol**2)*T + vol*(T**0.5)*Z)
    return max(K-S_T,0)

def eulerMethodCall(S,T,K,r,vol):
    Z = np.random.normal()
    S_T = S * np.exp((r-0.5*vol**2)*T + vol*(T**0.5)*Z)
    return max(S_T-K,0)

def N_func(x):
    return norm.cdf(x)

def black_scholes(vol, S, T, K, r):
    d1 = (np.log(S/K) + (r+0.5*vol**2)*T)/(vol*T**0.5)
    d2 = d1 - (vol*T**0.5)
    return - S * N_func(-d1) + np.exp(-r * T) * K * N_func(-d2)

def monte_Carlo(S,T,K,r,vol,factor_list,simulations):
    averages = []
    for factor in factor_list:
        N_samples = 10**factor
        for sim in range(simulations):
            samples = []
            for i in range(N_samples):
                sample = eulerMethodPut(S,T,K,r,vol)
                samples.append(sample)
            average = np.exp(-r*T)*np.mean(samples)
            std = np.std(samples)
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
simulations = 20

MC_list = monte_Carlo(S,T,K,r,vol,factor_list,simulations)
df = pd.DataFrame(MC_list,columns=["N_samples","value"])
df.to_csv("MC_normal.csv")

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

print("Analytical:", black_scholes(vol, S, T, K, r))

epsilon = [0.01, 0.02, 0.5]
N_samples = 1000000

for eps in epsilon:
    print(S+eps)
    samples = []
    np.random.seed(42)
    for i in range(N_samples):
        sample = eulerMethodPut(S+eps,T,K,r,vol)
        samples.append(sample)
    average = np.exp(-r*T)*np.mean(samples)
    print(average)

    samples2 = []
    np.random.seed(42)
    for i in range(N_samples):
        sample2 = eulerMethodPut(S,T,K,r,vol)
        samples2.append(sample2)
    average2 = np.exp(-r*T)*np.mean(samples2)
    print(average2)

    print("Procent:",((average-average2)/eps)/average*100)
    print("Dif:", (average-average2)/eps,"\n")


#%%
# 2) Digital option

N_samples = 100

payment = 0
payoff_list = []
np.random.seed(42)
for i in range(N_samples):
    payoff = eulerMethodPut(S,T,K,r,vol)
    if payoff != 0:
        payoff += 1
    payoff_list.append(payoff)

print(payoff_list)

# Pathwise method

