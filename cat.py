import numpy as np
import matplotlib.pyplot as plt

def buildTree(S, vol , T, N): 
    dt = T / N
    matrix = np.zeros((N + 1, N + 1))
    u = np.exp(vol*np.sqrt(dt))
    d = np.exp(-vol*np.sqrt(dt))
# Iterate over the lower triangle

    for i in np.arange(N + 1): # iterate over rows
        for j in np.arange(i + 1): # iterate over columns

# Hint: express each cell as a combination of up
# and down moves 
            matrix[i, j]=S * d**(i-j) * u**(j)
    return matrix

#sigma = 0.2 
#S = 100 
#T=1.
#N=50
#tree = buildTree(S, sigma, T, N)
#print(tree)

def valueOptionMatrix(tree , T, r , K, vol, N, option = "call"):
    dt = T / N
    u = np.exp(vol*np.sqrt(dt))
    d = np.exp(-vol*np.sqrt(dt))
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
S = 100 
T=1.
N=2
K = 99 
r = 0.06
tree = buildTree(S, sigma, T, N) 
#print(tree)
tree2 = valueOptionMatrix ( tree , T, r , K, sigma , N)
#print(tree2)


# Play around with different ranges of N and step sizes . 
N = 300
# Calculate the option price for the correct parameters 
# optionPriceAnalytical = np.exp(r*T) * S
optionPriceAnalytical = K-S # black-schioe
list_anal = [] 
list_approx = []
# calculate option price for each n in N 
for n in range(1,N):
    treeN = buildTree(S, sigma, T, n) 
    priceApproximatedly = valueOptionMatrix ( treeN , T, r , K, sigma, n )
    list_anal.append(optionPriceAnalytical)
    list_approx.append(priceApproximatedly[0,0])
# use matplotlib to plot the analytical value 
# and the approximated value for each n

#plt.plot(range(1,N), list_anal, label = "Analytical")
plt.plot(range(1,N), list_approx, label = "Approximatedly" )
plt.legend()
plt.show()
