"""
Assignment 1 Computational Finance
"""
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt

""" B - The Binomial Tree"""
def buildTree(S, vol, T, N):
    dt = T/N
    matrix = np.zeros((N+1, N+1))

    u = np.exp(vol * np.sqrt(dt))
    d = np.exp(-vol * np.sqrt(dt))

    for i in np.arange(N+1):
        for j in np.arange(i+1):
            value = (S*(d**(i-j)))*u**j
            matrix[i,j] = value
    return matrix


def valueOptionMatrix(tree, T, r, K, vol, N):
    dt = T/N
    u = np.exp(vol * np.sqrt(dt))
    d = np.exp(-vol * np.sqrt(dt))

    p = (np.exp(r*dt)-d)/(u-d)

    colomns = tree.shape[1]
    rows = tree.shape[0]

    # Walk backward, starting in last row of the matrix

    # Add the payoff functions in the last row
    for c in np.arange(colomns):
        S = tree[rows-1, c]
        tree[rows-1, c] = max(S-K,0)

    # For all other rows, we need to combine from previousrows
    # We walk backwards, from the last row to the first row
    for i in np.arange(rows-1)[::-1]:
        for j in np.arange(i+1):
            down = tree[i+1,j]
            up = tree[i+1,j+1]
            tree[i,j] = (p*up+(1-p)*down) * np.exp(-r*dt)
    
    return tree

vol = 0.2
S = 80
T = 1.

N = 50

K = 85
r = 0.1

tree = buildTree(S, vol, T, N)

def N(x):
    np.cfnd

errorList = []
analyticalList = []
approxList = []
for n in range(1,N+1):
    treeN = buildTree(S, vol, T, n)
    d1 = (np.log(S/K) + (r+0.5*vol**2)*T)/(vol*T**0.5)
    d2 = d1 - (vol*T**0.5)
    optionPriceAnalytical = S * N(d1) - np.exp(-r * T) * K * N(d2)
    priceApproximatedly = valueOptionMatrix(treeN, T, r, K, vol, n)[0,0]
    errorList.append(optionPriceAnalytical- priceApproximatedly)
    analyticalList.append(optionPriceAnalytical)
    approxList.append(priceApproximatedly)

plt.plot(range(1,N+1),errorList,label="error")
plt.plot(range(1,N+1),analyticalList, label = "analytical")
plt.plot(range(1,N+1),approxList, label="approx")
plt.xlabel("N")
plt.ylabel("Error")
plt.legend()
plt.show()


""" C - The Black Scholes Model """

# r = constant
# vol = constant
# z_t = brownian/wiener

# dSt = r * St * dt + vol * St * z_t

""" REAAAAAL ASSIGNMENT """
# Part 1.1

