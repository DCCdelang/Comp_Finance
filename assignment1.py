import numpy as np
import matplotlib.pyplot as plt
from math import *
from scipy.stats import norm


def buildTree(S,sigma,T,N):
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




def valueOptionMatrix( tree , T, r , K, sigma,N ) :
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
sigma = 0.2
S = 99
T = 1.
N = 50
K = 100
r = 0.06


def N_(x):
    #'Cumulative distribution function for the standard normal distribution'

    # print((1.0 + erf(x / sqrt(2.0))) / 2.0)
    
    return norm.cdf(x)

def black_scholes(S,N,T,sigma,r,K):
    d = ((np.log(S/K) + ((r-(sigma**2)/2)) * T )/(sigma*np.sqrt(T)))
    d_1 = d + sigma* np.sqrt(T)
    return (S*N_(d_1)) - (K*np.exp(-r*T)*N_(d))

def convergence(S,N,T,sigma,r,K):
    y = []
    z = []

    # Calculate the option price for the correct parameters
    optionPriceAnalytical = black_scholes(S,N,T,sigma,r,K)
    
    # calculate option price for each n in N
    for n in range(1, N):
        treeN = buildTree(S,sigma,T,n) 
        priceApproximatedly = valueOptionMatrix(treeN,T,r,K,sigma,n)[0][0]
        y.append(priceApproximatedly)
        z.append(optionPriceAnalytical)


    # use matplotlib to plot the analytical value
    # and the approximated value for each n
    plt.plot(range(N-1), z)
    plt.plot(range(N-1), y)
    plt.show()

def sigma_change(S,N,T,sigmas,r,K):
    y_ = []

    for sigma in sigmas:
        # Calculate the option price for the correct parameters
        optionPriceAnalytical = black_scholes(S,N,T,sigma,r,K)
        print(sigma)
        
        # calculate option price for each n in N
        for n in range(1, N):
            print(sigma)
            treeN = buildTree(S,sigma,T,n) 
            priceApproximatedly = valueOptionMatrix(treeN,T,r,K,sigma,n)[0][0]
        y_.append(abs(priceApproximatedly - optionPriceAnalytical))
    plt.plot(sigmas, y_)


    plt.show()

sigma_change(S,N,T,sigmas,r,K)
convergence(S,N,T,sigma,r,K)