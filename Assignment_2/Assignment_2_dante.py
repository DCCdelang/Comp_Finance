"""
Assignment 2 Computational Finance
Author: Dante de Lang
"""
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
rs = RandomState(MT19937(SeedSequence(123456789)))

def eulerMethodPut(S,T,K,r,vol):
    Z = np.random.normal()
    S_T = S * np.exp((r-0.5*vol**2)*T + vol*(T**0.5)*Z)
    return max(K-S_T,0)

vol = 0.2
K = 99
S = 100
r = 0.06
T = 1

factor_list = [1]

for factor in factor_list:
    N_samples = 1000 * factor
    samples = []
    for i in range(N_samples):
        sample = eulerMethodPut(S,T,K,r,vol)
        if sample != 0:
            samples.append(sample)

    sns.histplot(samples)
plt.show()
