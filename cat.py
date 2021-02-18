#%%
import numpy as np
import matplotlib.pyplot as plt
from math import *
from scipy.stats import norm

def buildTree(S, sigma , T, N): 
    dt = T / N
    matrix = np.zeros((N + 1, N + 1))
    u = np.exp(sigma*np.sqrt(dt))
    d = np.exp(-sigma*np.sqrt(dt))

    for i in np.arange(N + 1): # iterate over rows
        for j in np.arange(i + 1): # iterate over columns
            matrix[i, j]=S * d**(i-j) * u**(j)
    return matrix

#sigma = 0.2 
#S = 100 
#T=1.
#N=50
#tree = buildTree(S, sigma, T, N)
#print(tree)

def valueOptionMatrix(S, T, r , K, sigma, N, option = "call"):
    tree = buildTree(S, sigma, T, N)
    dt = T / N
    u = np.exp(sigma*np.sqrt(dt))
    d = np.exp(-sigma*np.sqrt(dt))
    p = (np.exp(r*dt) - d)/(u - d)
    columns = tree.shape[1] 
    rows = tree.shape[0]
# Walk backward , we start in last row of the matrix
# Add the payoff function in the last row 
    for c in np.arange(columns):
        S = tree[rows-1, c] # value in the matrix
        if (option == "call"): 
            tree[rows-1 , c ] = max(S-K, 0)
        elif (option == "put"):
            tree[rows-1 , c ] = max(K-S, 0)
# For all other rows , we need to combine from previous rows 
# We walk backwards , from the last row to the first row
    #print(tree)
    for i in np.arange(rows-1)[::-1]: 
        for j in np.arange(i + 1):
            down = tree[i + 1, j] 
            up = tree[i + 1, j + 1]
            tree[i,j] = (p*up + (1-p)*down) * np.exp(-r*dt)
    return tree


sigma = 0.2
S = 50
T=2.
K = 52 
r = 0.05
N = 2
#tree = valueOptionMatrix ( tree , T, r , K, sigma , N)
#print(tree)


### 1.2: Calculate the option price in the analytical and approximated way for different values
#        of volatility
def black_scholes(S,N,T,sigma,r,K):
    d = ((np.log(S/K) + ((r-(sigma**2)/2)) * T )/(sigma*np.sqrt(T)))
    d_1 = d + sigma* np.sqrt(T)
    return (S*norm.cdf(d_1)) - (K*np.exp(-r*T)*norm.cdf(d))

sigma_list = np.linspace(0.01, 10, 400)
S = 99
T=1.
K = 100 
r = 0.06
N = 50
error = []

for sigma in sigma_list:
    list_anal = [] 
    list_approx = []
    optionPriceAnalytical = black_scholes(S, N, T, sigma, r, K)
    for n in range(1,N+1):
        priceApproximatedly = valueOptionMatrix (S, T, r , K, sigma, n )
        list_anal.append(optionPriceAnalytical)
        list_approx.append(priceApproximatedly[0,0])
    error.append(abs(list_anal[-1]-list_approx[-1]))

plt.plot(sigma_list, error)
plt.xlabel("volatility")
plt.ylabel("Option error")
plt.show()
"""

    plt.plot(range(1,N), list_anal, label = "Analytical")
    plt.plot(range(1,N), list_approx, label = "Approximatedly" )
    plt.text(45, 11.55, f"$\sigma$ = {sigma}")
    plt.legend()
    plt.show()
"""


### 1.3: convergence of the method for increasing number of steps in the tree
S = 99
T=1.
K = 100 
r = 0.06
N = 50
sigma = 0.2
error = []


for n in range(1,N):
    priceApproximatedly = valueOptionMatrix(S, T, r , K, sigma, n )
    anal = black_scholes(S, N, T, sigma, r, K)
    approx = priceApproximatedly[0,0]
    error.append(abs(anal-approx))

plt.plot(range(1,N), error)
plt.xlabel("N")
plt.ylabel("error")
plt.show()

# computational complexity
n = np.linspace(1,N, N-1)
plt.plot(n, 1/2*n**2)
plt.xlabel("N")
plt.ylabel("computational complexity")
plt.show()

# %%
### Comparison for delta
def calc_delta(S, T, r , K, sigma, N, option = "call"):
    dt = T / N
    u = np.exp(sigma*np.sqrt(dt))
    d = np.exp(-sigma*np.sqrt(dt))
    p = (np.exp(r*dt) - d)/(u - d)
    tree_payoff = valueOptionMatrix(S, T, r , K, sigma, N, option)
    delta = (tree_payoff[1, 1]- tree_payoff[1,0])/(S*u-S*d)
    #print(delta)
    #if (option == "call" & delta < 0):
    #    print("Something went wrong, delta should be positive")
    #if (option == "put" & delta > 0):
    #    print("Something went wrong, delta should be negative")
    
    return delta

sigma_list = np.linspace(0.05, 10, 400)
S = 99
T=1.
K = 100 
r = 0.06
N = 50
error = []

for sigma in sigma_list:
    delta = calc_delta(S, T, r , K, sigma, N, option = "call")
    d = ((np.log(S/K) + ((r+(sigma**2)/2)) * T )/(sigma*np.sqrt(T)))
    d_1 = d - sigma* np.sqrt(T)
    delta_BS = norm.cdf(d_1)
    error.append(abs(delta-delta_BS))
#print(error)
plt.plot(sigma_list, error)
plt.xlabel("volatility")
plt.ylabel("Delta error")
plt.show()
# %%
