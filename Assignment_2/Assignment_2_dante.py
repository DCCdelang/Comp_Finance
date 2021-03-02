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

def eulerMethodPut(S,T,K,r,vol):
    Z = np.random.normal()
    S_T = S * np.exp((r-0.5*vol**2)*T + vol*(T**0.5)*Z)
    return max(K-S_T,0)

def N_func(x):
    return norm.cdf(x)

def black_scholes(vol, S, T, K, r):
    d1 = (np.log(S/K) + (r+0.5*vol**2)*T)/(vol*T**0.5)
    d2 = d1 - (vol*T**0.5)
    return - S * N_func(-d1) + np.exp(-r * T) * K * N_func(-d2)

vol = 0.2
K = 99
S = 100
r = 0.06
T = 1

BS_value = black_scholes(vol, S, T, K, r)
print("Analytical value", BS_value)

factor_list = [2,3,4,5]
simulations = 3

averages = []

for K in np.arange(70,99,5):
    for factor in factor_list:
        start = time.time()
        N_samples = 10**factor
        for sim in range(simulations):
            samples = []
            for i in range(N_samples):
                sample = eulerMethodPut(S,T,K,r,vol)
                if sample != 0:
                    samples.append(sample)
            average = np.exp(-r*T)*(sum(samples)/N_samples)
            averages.append([N_samples,average])
        print(N_samples, "takes", time.time()-start, "seconds")

    df = pd.DataFrame(averages,columns=["N_samples","value"])
    sns.lineplot(data=df,x="N_samples",y="value", label="MC"+str(K))
# plt.hlines(BS_value,averages[0][0],averages[-1][0],label="BS")
plt.xscale("log")
plt.legend()
plt.show()
# plt.pause(5)
# plt.close()