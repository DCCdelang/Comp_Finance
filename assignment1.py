import numpy as np
import matplotlib.pyplot as plt
from math import *
from scipy.stats import norm


def buildTree(S,vol,T,N):
    dt=T/N
    matrix=np.zeros((N+1,N+1))
    
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp( - sigma * np.sqrt(dt))

    #Iterate over the lower triangle
    for i in np.arange (N + 1 ) : # iterateoverrows
        for j in np.arange ( i + 1 ) :
            A = S
            for _ in range(i):
                A = A * d  
            for _ in range(j*2):
                A = A * u
            matrix[i,j] = A
    # print(matrix)        
    return matrix




def valueOptionMatrix( tree , T, r , K, vol,N ) :
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp( - sigma * np.sqrt(dt))
    p = (np.exp(r * dt) - d)/(u-d)
    columns = tree.shape [ 1 ]
    rows = tree.shape [ 0 ]
    # Walk backward , we start in last row of the matrix
    # Add the payoff function in the last row

    # print(columns)
    for c in np.arange(columns):
        S = tree[rows - 1 , c] # value in the matr ix
        pay_off = max(S-K, 0)
        tree[ rows - 1 , c] = pay_off
  
    # For all other rows , we need to combine from previous rows
    # We walk backwards , from the last row to the first row
    for i in np.arange ( rows - 1 ) [ : : - 1 ]:
        for j in np.arange( i + 1 ) :

            # print(i,j)
            down = tree[ i + 1 , j ]
            up = tree[ i + 1 , j + 1 ]

            tree[i, j] = (p*up + (1-p)*down) * np.exp(-r*dt)
            

    # print(tree)
    return tree



sigmas = np.linspace(0.05, 0.95, 19)
print(sigmas)
S = 99
T = 1.
N = 2
K = 100
r = 0.06

# tree = buildTree(S,sigma,T,N)
# valueOptionMatrix( tree , T, r , K, sigma )

def N_(x):
    #'Cumulative distribution function for the standard normal distribution'
    print((1.0 + erf(x / sqrt(2.0))) / 2.0)
    print(norm.cdf(x))
    return (1.0 + erf(x / sqrt(2.0))) / 2.0

def black_scholes(S,N,T,sigma,r,K):
    # a = np.log(S/K)
    # b = (r-sigma**2)/2
    # c = sigma * np.sqrt(T)

    # d = (a+b*T)/c
    # d_1 = d + c
    # print(a,b,c)
    d = ((np.log(S/K) + ((r-(sigma**2)/2)) * T )/(sigma*np.sqrt(T)))
    d_1 = d + sigma* np.sqrt(T)
    return (S*N_(d_1)) - (K*np.exp(-r*T)*N_(d))

# Play around with different ranges of N and step sizes .
y_ = []
z_ = []
for sigma in sigmas:
    y = []
    z = []
    X = 51
    # Calculate the option price for the correct parameters
    optionPriceAnalytical = black_scholes(S,N,T,sigma,r,K)
    # raise ValueError()
    # print(optionPriceAnalytical)
    
    # calculate option price for each n in N
    for n in range(1, X):
        # print(n)
        treeN = buildTree(S,sigma,T,n) 
        priceApproximatedly = valueOptionMatrix(treeN,T,r,K,sigma,n)[0][0]
        y.append(priceApproximatedly)
        z.append(optionPriceAnalytical)
    y_.append(abs(priceApproximatedly - optionPriceAnalytical))
        
        
    # use matplotlib to plot the analytical value
    # and the approximated value for each n
    plt.plot(range(X-1), z)
    plt.plot(range(X-1), y)
    plt.show()

# print(z[-1], y[-1])
# plt.plot(sigmas, z_)
plt.plot(sigmas, y_)


plt.show()