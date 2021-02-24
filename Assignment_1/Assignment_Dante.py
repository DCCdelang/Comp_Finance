#%%
"""
Assignment 1 Computational Finance
"""
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
rs = RandomState(MT19937(SeedSequence(123456789)))

#%%
""" 
Part 1.1: Price approximation
"""
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


def valueOptionMatrix(tree, T, r, K, vol, N, option="Call", type_option="European"):
    dt = T/N
    u = np.exp(vol * np.sqrt(dt))
    d = np.exp(-vol * np.sqrt(dt))

    p = (np.exp(r*dt)-d)/(u-d)

    colomns = tree.shape[1]
    rows = tree.shape[0]
    tree_copy = tree.copy()
    # Walk backward, starting in last row of the matrix

    # Add the payoff functions in the last row
    for c in np.arange(colomns):
        S = tree[rows-1, c]
        if option == "Call":
            tree[rows-1, c] = max(S-K,0)
        elif option == "Put":
            tree[rows-1, c] = max(K-S,0)

    # For all other rows, we need to combine from previousrows
    # We walk backwards, from the last row to the first row
    for i in np.arange(rows-1)[::-1]:
        for j in np.arange(i+1):
            down = tree[i+1,j]
            up = tree[i+1,j+1]
            if type_option == "European":
                tree[i,j] = (p*up+(1-p)*down) * np.exp(-r*dt)
            elif type_option == "American":
                payoff1 = (p*up+(1-p)*down) * np.exp(-r*dt)
                if option == "Put":
                    payoff2 = K-tree_copy[i,j]
                    tree[i,j] = max(payoff1,payoff2)
                elif option == "Call":
                    payoff2 = tree_copy[i,j]-K
                    tree[i,j] = max(payoff1,payoff2)
    
    
    return tree

S = 99
T = 1
r = 0.06
K = 100
vol = 0.2
N = 50

tree = buildTree(S, vol, T, N)
approx_option_price = valueOptionMatrix(tree, T, r, K, vol, N)[0,0]
print("For 1.1 option price approximation is", approx_option_price)
#%%
""" 
Part 1.2: Comparison of option price approximation to black scholes, different
volatilities.
"""

S = 100
T = 1
r = 0.06
K = 99
vol = 0.2
N = 50

def N_func(x):
    return norm.cdf(x)

def black_scholes(vol, S, T, K, r):
    d1 = (np.log(S/K) + (r+0.5*vol**2)*T)/(vol*T**0.5)
    d2 = d1 - (vol*T**0.5)
    return S * N_func(d1) - np.exp(-r * T) * K * N_func(d2)

# Convergence for different volatility values
volList = [0.2, 0.5, 0.95]
N = 50
for vol in volList:
    # errorList = []
    analyticalList = []
    approxList = []
    for n in range(1,N+1):
        treeN = buildTree(S, vol, T, n)
        priceApproximatedly = valueOptionMatrix(treeN, T, r, K, vol, n)[0,0]
        optionPriceAnalytical = black_scholes(vol, S, T, K, r)

        # errorList.append(optionPriceAnalytical- priceApproximatedly)
        analyticalList.append(optionPriceAnalytical)
        approxList.append(priceApproximatedly)

    plt.plot(range(1,N+1),analyticalList,"k")
    plt.plot(range(1,N+1),approxList, label="vol = "+str(round(vol,2)))
plt.xlabel("N",fontsize=14)
plt.ylabel("Value",fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.legend(loc=1,fontsize=14)
plt.tight_layout()
plt.savefig("compare_bs-bin2.pdf")
plt.show()

# Volatility vs error
volList = np.linspace(0.01,10,400)
errorList = []
for vol in volList:
    treeN = buildTree(S, vol, T, N)
    priceApproximatedly = valueOptionMatrix(treeN, T, r, K, vol, N)[0,0]
    optionPriceAnalytical = black_scholes(vol, S, T, K, r)
    errorList.append(abs(optionPriceAnalytical - priceApproximatedly))

plt.plot(volList,errorList,label="error "+str(round(vol,2)))
plt.xlabel("Volatility",fontsize=14)
plt.ylabel("Error",fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.legend(fontsize=14)
# plt.legend(loc=1)
plt.tight_layout()
plt.savefig("error_bs-bin.pdf")
plt.show()
#%%
""" 
Part 1.3: Study convergence N
# """
N = 200
analyticalList = []
approxList = []
for n in range(1,N+1):
    treeN = buildTree(S, vol, T, n)
    priceApproximatedly = valueOptionMatrix(treeN, T, r, K, vol, n)[0,0]
    optionPriceAnalytical = black_scholes(vol, S, T, K, r)

    # errorList.append(optionPriceAnalytical- priceApproximatedly)
    analyticalList.append(optionPriceAnalytical)
    approxList.append(priceApproximatedly)

plt.plot(range(1,N+1),analyticalList)
plt.plot(range(1,N+1),approxList, label="N "+str(round(N,2)))
plt.xlabel("N")
plt.ylabel("Value")
plt.legend(loc=1)
plt.show()

# Computational complexity for the binomial tree is 1/2*N^2 looking at the
# for loops presented in the buildTree() function --> O(N^2)

"""
Part 1.4: Delta parameter
"""
# When C is derived over S the only term standing is N(d_1)
# See finance paper page 9 --> Full derivation still needed

#%%
"""
Part 1.5: Compute hedge parameter
"""
N = 50

delta_error_list,delta_bs_list, delta_binom_list = [],[],[]
volList = np.linspace(0.01,10,100)
for vol in volList:
    treeN = buildTree(S, vol, T, N)
    delta_S = treeN[1,0]-treeN[1,1]

    priceApproximatedly = valueOptionMatrix(treeN, T, r, K, vol, N)
    delta_f = priceApproximatedly[1,0]-priceApproximatedly[1,1]
    delta_binom = delta_f / delta_S
    delta_bs = N_func(((np.log(S/K) + (r+0.5*vol**2)*T)/(vol*T**0.5))) # Which one is correct? 

    delta_bs_list.append(delta_bs)
    delta_binom_list.append(delta_binom)
    delta_error_list.append(abs(delta_binom-delta_bs))

# plt.plot(volList,delta_error_list,label="error")
plt.plot(volList,delta_bs_list,label="black scholes")
plt.plot(volList,delta_binom_list,label="binom")
plt.xlabel("Volatility")
plt.ylabel("Fraction")
plt.legend()
plt.show()
plt.plot(volList,delta_error_list)
plt.xlabel("Volatility")
plt.ylabel("Error")
plt.show()

#%%
"""
Part 1.6: American instead of European
"""
S = 100
T = 1
r = 0.06
K = 99
vol = 0.2
N = 50



# treeN = buildTree(S, vol, T, N)
# approx_matrix = valueOptionMatrix(treeN, T, r, K, vol, N, option="Put", type_option="European")
# print("European Put:",approx_matrix[0,0])

volList = [0.2]
ep, ec, ap, ac = [],[],[],[]
for vol in volList:
    treeN = buildTree(S, vol, T, N)
    ep.append(valueOptionMatrix(treeN, T, r, K, vol, N, option="Put", type_option="European")[0,0])
    treeN = buildTree(S, vol, T, N)
    ec.append(valueOptionMatrix(treeN, T, r, K, vol, N, option="Call", type_option="European")[0,0])
    treeN = buildTree(S, vol, T, N)
    ap.append(valueOptionMatrix(treeN, T, r, K, vol, N, option="Put", type_option="American")[0,0])
    treeN = buildTree(S, vol, T, N)
    ac.append(valueOptionMatrix(treeN, T, r, K, vol, N, option="Call", type_option="American")[0,0])
print(ep)
raise ValueError()
plt.plot(volList,ep,label="EU-Put")
plt.plot(volList,ec,label="EU-Call")
plt.plot(volList,ap,"--",label="AM-Put")
plt.plot(volList,ac,"--",label="AM-Call")
plt.xlabel("Volatility",fontsize=14)
plt.ylabel("Option price",fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig("american.pdf")
plt.show()

N=50
ep, ec, ap, ac = [],[],[],[]
for N in range(1,N+1):
    treeN = buildTree(S, vol, T, N)
    ep.append(valueOptionMatrix(treeN, T, r, K, vol, N, option="Put", type_option="European")[0,0])
    treeN = buildTree(S, vol, T, N)
    ec.append(valueOptionMatrix(treeN, T, r, K, vol, N, option="Call", type_option="European")[0,0])
    treeN = buildTree(S, vol, T, N)
    ap.append(valueOptionMatrix(treeN, T, r, K, vol, N, option="Put", type_option="American")[0,0])
    treeN = buildTree(S, vol, T, N)
    ac.append(valueOptionMatrix(treeN, T, r, K, vol, N, option="Call", type_option="American")[0,0])

plt.plot(range(1,N+1),ep,label="EU-Put")
plt.plot(range(1,N+1),ec,label="EU-Call")
plt.plot(range(1,N+1),ap,label="AM-Put")
plt.plot(range(1,N+1),ac,label="AM-Call")

plt.xlabel("N",fontsize=14)
plt.ylabel("Value",fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.legend(loc=1,fontsize=14)
plt.tight_layout()
plt.savefig("compare_bs-bin2.pdf")
plt.show()

# %%
"""
Part 2.1: Derive risk neutral pricing formula
"""
# Derivation is in send pdf
#%%
"""
Part 2.2: Euler method to perform a hedging simulation
"""

vol = 0.2
K = 99
S = 100
r = 0.06
N = 365
T = 1
# Frequency is between daily and weekly

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

def eulerApproxMethod(S,T,N,M,r,vol):
    dt = T/N
    S_list = []
    S_list1 = []
    Sm, Sm1 = S, S
    for n in range(N):
        Zm = np.random.normal()
        Snext = Sm + r*Sm*dt + vol*Sm*(dt**0.5)*Zm
        Sm = Snext
        if n % M == 0:
            S_list.append(Sm)
        else:
            S_list1.append(Sm)
    return S_list, S_list1


vol = 0.2
np.random.seed(42)
exactList = exactMethod(S,T,N,r,vol)
x = np.linspace(0,365,N+1)
plt.plot(x,exactList,label="Exact")

vol = 0.1
np.random.seed(42)
M = 21
approxList1,approxList2 = eulerApproxMethod(S,T,N,M,r,vol)
x = np.linspace(0,365,len(approxList2))
plt.plot(x,approxList2,label=f"Vol={round(vol,2)}")

volList = [0.5,1,2,4.5]
for vol in volList:
    np.random.seed(42)
    M = 21
    approxList1,approxList2 = eulerApproxMethod(S,T,N,M,r,vol)

    # x = np.linspace(0,365,len(approxList1))
    # plt.plot(x,approxList1)

    x = np.linspace(0,365,len(approxList2))
    plt.plot(x,approxList2,label=f"Vol={round(vol,2)}")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.xlabel("N",fontsize=14)
plt.ylabel("Price",fontsize=14)
plt.tight_layout()
plt.savefig("euler_vol.pdf")
plt.show()

# %%
"""
Complexity binomial tree
"""

def buildTreeComp(S, vol, T, N):
    dt = T/N
    matrix = np.zeros((N+1, N+1))

    u = np.exp(vol * np.sqrt(dt))
    d = np.exp(-vol * np.sqrt(dt))

    complexity = []
    for i in np.arange(N+1):
        comp = 0
        for j in np.arange(i+1):
            comp += j
            # value = (S*(d**(i-j)))*u**j
            # matrix[i,j] = value
        complexity.append(comp)
    return complexity

N = 50
complexity_list = buildTreeComp(S, vol, T, N)
plt.plot(range(N+1),complexity_list)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.legend(fontsize=14)
plt.xlabel("N",fontsize=14)
plt.ylabel("Computational complexity",fontsize=14)
plt.tight_layout()
plt.savefig("complexity.pdf")
plt.show()