import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import *
from scipy.stats import norm
import seaborn as sns


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



sigmas = np.linspace(0.01, 10, 200)
Ns = np.linspace(5,50,10)
sigma = 0.2
S = 99
T = 1.
N = 50
K = 100
r = 0.06
# print(buildTree(S,sigma,T,N))
# raise ValueError()

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
    an = []
    bi = []
    for sigma in sigmas:
        # Calculate the option price for the correct parameters
        optionPriceAnalytical = black_scholes(S,N,T,sigma,r,K)
        
        # calculate option price for each n in N

        treeN = buildTree(S,sigma,T,N) 
        priceApproximatedly = valueOptionMatrix(treeN,T,r,K,sigma,N)[0][0]
        y_.append(abs(optionPriceAnalytical - priceApproximatedly ))
        an.append(optionPriceAnalytical)
        bi.append(priceApproximatedly)
    plt.plot(sigmas, an, label="an")
    plt.plot(sigmas, bi, label = "bi")

    plt.plot(sigmas, y_)
    # plt.yscale('log')
    plt.legend()


    plt.show()


def N_change(S,Ns,T,sigmas,r,K):
    y_ = []

    for N in Ns:
        # Calculate the option price for the correct parameters
        optionPriceAnalytical = black_scholes(S,N,T,sigma,r,K)
        print(N)
        
        # calculate option price for each n in N
        for n in range(1, int(N)):
            # print(sigma)
            treeN = buildTree(S,sigma,T,n) 
            priceApproximatedly = valueOptionMatrix(treeN,T,r,K,sigma,n)[0][0]
        y_.append(abs(priceApproximatedly - optionPriceAnalytical))
    plt.plot(Ns, y_)

    plt.xlabel("N", fontsize=14)
    plt.ylabel("Error", fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.tight_layout()
    plt.savefig("N_error.pdf")
    plt.show()

def Hedge(S,N,T,sigmas,r,K):
    l = []
    b = []
    a = []
    for sigma in sigmas:
        tree = buildTree(S,sigma,T,N)
        low = tree[1, 0]
        high = tree[1, 1]
        # print(high, low)
        options = valueOptionMatrix( tree , T, r , K, sigma,N )
        low_option = options[1,0]
        high_option = options[1,1]

        delta = (high_option - low_option)/(high - low)
        d = ((np.log(S/K) + ((r+(sigma**2)/2)) * T )/(sigma*np.sqrt(T)))
        d_1 = d - sigma* np.sqrt(T)
        d_s = N_(d)
        a.append(d_s)
        b.append(delta)
        l.append(abs(delta - d_s))
        print(delta)
    plt.plot(sigmas, a, label="Black-Scholes")
    plt.plot(sigmas,b,"--", label="Binomial")
    plt.xlabel("Volatility", fontsize=14)
    plt.ylabel("Fraction", fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.savefig("Hedge_parameter.pdf")
    plt.show()
    plt.plot(sigmas, l)
    plt.xlabel("Volatility", fontsize=14)
    plt.ylabel("Difference", fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    # plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig("Hedge_difference.pdf")
    plt.show()


def exactMethod(S,T,N,r,vol):
    dt = T/N
    S_list = [S]
    Sm = S
    for _ in range(N):
        Zm = np.random.normal()
        Snext = Sm * np.exp((r-0.5*(vol**2))*dt+vol*(dt**0.5)*Zm)
        Sm = Snext
        S_list.append(Sm)
    return S_list

def eulerApproxMethod(S,T,N,r,vol,hedge="both"):
    dt = T/N
    A = []
    B = []
    S_list = [S]
    Sm = S
    for n in range(N):
        Zm = np.random.normal()
        if n % 7 == 0:
            Snext = Sm + r*Sm*dt + vol*Sm*(dt**0.5)*Zm
            Sm = Snext
            A.append(Sm)
            B.append(Sm)
        else:
            Snext = Sm + r*Sm*dt + vol*Sm*(dt**0.5)*Zm
            Sm = Snext
            B.append(Sm)
    return A, B



time = []
value = []


# exactList = exactMethod(S,T,N,r,vol)
# x = np.linspace(0,365,N+1)
# plt.plot(x,exactList, label="Exact")

# for i in range(10):
#     approxList = eulerApproxMethod(S,T,N,r,sigma)[1]
#     count = 0
#     for i in approxList:
#         value.append(i)
#         time.append(count)
#         count +=1
for i in range(1000000):
    approxList = eulerApproxMethod(S,T,N,r,sigma)[1][-1]
    count = 0
    value.append(approxList)
    time.append(count)
    count +=1

data = {"Values":value}
df = pd.DataFrame(data) 

sns.histplot(df, x="Values", kde=True)

# sns.lineplot(data=df, x="Time", y="Values")

plt.show()
# approxList = eulerApproxMethod(S,T,N,r,vol)[0]
# x = np.linspace(0,365,len(approxList))
# plt.plot(x,approxList,":" ,label="Euler-weekly")
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.xlabel("Time", fontsize=14)
# plt.ylabel("Stock price", fontsize=14)
# plt.legend(fontsize=14)
# plt.savefig("Eurler_same_vol.pdf")

plt.show()


# raise ValueError()
# Hedge(100,N,T,[0.2],r,99)
# sigma_change(S,N,T,sigmas,r,K)
# convergence(S,N,T,sigma,r,K)
# N_change(S,Ns,T,0.2,r,K)