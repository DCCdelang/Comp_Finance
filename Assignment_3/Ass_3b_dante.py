import numpy as np
import scipy.sparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def bottom_boundary_condition(t):
    return np.zeros(t.shape)

def top_boundary_condition(K,T,S_max, r, t):
    return S_max-np.exp(-r*(T-t))*K
    
def final_boundary_condition(K,S):
    return np.maximum(S-K,0)

def compute_abc( K, T, sigma, r, S, dt, dS ):
    dX = S/dS
    dX2 = S**2/dS**2

    # Solution from cats friend
    # k = r * 0.5 * 1/dX - sigma**2 * 0.5 * 1/dX
    # lam = 0.5 * sigma**2 * 1/dX2
    # mu = r
    # a = (k + lam)
    # b = (2 * lam - mu)
    # c = (-k + lam)

    # solution from original code: https://towardsdatascience.com/option-pricing-using-the-black-scholes-model-without-the-formula-e5c002771e2f
    # a = -sigma**2 * S**2/(2* dS**2 ) + r*S/(2*dS)
    # b = r + sigma**2 * S**2/(dS**2)
    # c = -sigma**2 * S**2/(2* dS**2 ) - r*S/(2*dS)

    # Based on own derivations
    a = 0.5 * sigma**2 / dX2 - (r-0.5 * sigma**2)/2*dX  # a-1
    b = - sigma**2/dX2 - r                              # a0
    c = 0.5 * sigma**2 / dX2 + (r-0.5 * sigma**2)/2*dX  # a+1

    return a,b,c

def compute_abc_CN( K, T, sigma, r, S, dt, dS ):
    dX = S/dS
    dX2 = S**2/dS**2

    # Based on own derivations
    a = 0.25 * sigma**2 / dX2 - (r-0.5 * sigma**2)/4*dX
    b = - sigma**2/2*dX2 - r/2
    c = 0.25 * sigma**2 / dX2 + (r-0.5 * sigma**2)/4*dX

    return a,b,c

def compute_try( S, T, sigma, r, dS, dt):
    dX = dS
    dX2 = dS**2
    # alpha = 2.0
    # # Calculated params (\Delta t should obey the FTCS condition for stability)
    # dt = (dX ** 2)/(4 * alpha)

    alpha = dt *(0.5 * sigma**2 / dX2 - (r-0.5 * sigma**2)/2*dX)  # a-1
    beta = 1 - dt * (- sigma**2/dX2 - r)                              # a0
    gamma = dt * (0.5 * sigma**2 / dX2 + (r-0.5 * sigma**2)/2*dX)

    return alpha, beta, gamma

def compute_matrix( a,b,c ):
    return scipy.sparse.diags( [a[1:],b,c[:-1]],offsets=[-1,0,1])

def compute_W(a,b,c, V0, VM): 
    M = len(b)+1
    W = np.zeros(M-1)
    W[0] = a[0]*V0 
    W[-1] = c[-1]*VM 
    return W

# def compute_k(gamma, N, t)
   
def price_call_explicit(S, K, T, r, sigma, N, M, CN = False):
    # Choose the shape of the grid
    dt = T/N
    S_min=0
    S0 = S
    S_max= S*(1+sigma)
    dS = (S_max-S_min)/M
    S = np.linspace(S_min,S_max,M+2)
    t = np.linspace(0,T,N)
    V = np.zeros((M+2,N+1)) #...
    
    if CN == False:
        # Set the boundary conditions
        V[-1,1:] = S_max-np.exp(-r*(T-t))*K
        V[:,0] = np.maximum(S-K,0)
        # V[-1,:] = np.maximum(S-K,0)

        # Setup boundaries on first column and last row
        # grid[:, 0] = np.maximum(np.exp(S) - K, 0)
        # grid[-1, 1:] = (np.exp(M2) - K) * np.exp(-r * tau_index * delta_tau)

        print(V)
        print(np.maximum(S-K,0))

        # Apply the recurrence relation
        a,b,c = compute_abc(K,T,sigma,r,S[1:-1],dt,dS)
        alpha, beta, gamma = compute_try(S,T,sigma,r, dS, dt)
        A = np.diag([alpha for i in range(M-1)], -1) + \
            np.diag([beta for i in range(M)]) + \
            np.diag([gamma for i in range(M-1)], 1)
        B = scipy.sparse.identity(M)
        B_inv = B

        # for n in range(0, N-1):
            # V[n+1,1:M] =  (identity-Lambda*dt).dot( V[n,1:M] )
        # alpha, beta, gamma = compute_try(K,T,sigma,r, dS, dt)
        
        # for n in range(0, N-1):
        #     for i in range(0,M):
        #         V[n+1,i] = alpha * V[n,i-1] + beta * V[n,i] + gamma * V[n,i+1]

        # for i in range(N,0,-1):
        #     W = compute_W(a,b,c,V[i,0],V[i,M])
        #     # Use `dot` to multiply a vector by a sparse matrix
        #     V[i-1,1:M] = (identity-Lambda*dt).dot( V[i,1:M] ) - W*dt

    elif CN == True:
        # Apply the recurrence relation
        a,b,c = compute_abc_CN(K,T,sigma,r,S[1:-1],dt,dS)

        # Set the boundary conditions
        V[:,0] -= 2*a[0]
        V[:,1] += a[0]
        V[-1,:] -= 2*c[-1]
        # V[-1,-2] += c[-1]

        Lambda1 = compute_lambda( a,b,c) 
        Lambda2 = compute_lambda( -a,-b,-c) 
        identity = scipy.sparse.identity(M-1)
        
        for i in range(N,0,-1):
            W = compute_W(a,b,c,V[i,0],V[i,M])
            # Use `dot` to multiply a vector by a sparse matrix
            V[i-1,1:M] = np.dot(((identity-Lambda2*dt).dot( V[i,1:M] ) - W*dt),scipy.linalg.inv((identity-Lambda1*dt).todense()))
    
    tau_index = np.arange(1, N+1)
    for t1 in tau_index:
        # Get V^n from grid
        V_prev = V[1:-1,t1-1]
        
        if CN == False:
            # k_1 is 0
            k_new = np.zeros(M)
            
            # Construct k_2
            k_prev = np.zeros(M)
            k_prev[0] = 0
            k_prev[-1] = gamma * V[t1-1,-1]
        else:
            # Construct k_1
            k_new = np.zeros(M)
            k_new[0] = 0
            k_new[-1] = gamma * V[t1,-1]
            
            # Construct k_2
            k_prev = np.zeros(M)
            k_prev[0] = 0
            k_prev[-1] = -gamma * V[t1-1, -1]

        # print(V_prev.shape)
        # print(A.shape)
        # print(B_inv.shape)
        # print(k_prev.shape)
        # print(k_new.shape)
        V[1:-1, t1] = B_inv.dot(A.dot(V_prev) + k_prev - k_new)

    # print(S[:-1].shape)
    # print(V[:-1, -1].shape)
    # opt_value = np.interp(S0, S[:-1], V[1:-1, -1])
    # print(opt_value)

    return V[:,:-1], t, S

sigma = 0.3
r = 0.04
S = 100
K = 110
T = 1
N = 1000 # t
M = 1000 # S

V, t, S = price_call_explicit(S, K, T, r, sigma, N, M, CN = False)

# print(V.shape)
# print(S,V)

from matplotlib import cm 

X, Y = np.meshgrid(t,S)
print("X.shape is: ", X.shape)
print("Y.shape is: ", Y.shape)
print("V.shape is: ", V.shape)


# Plot S
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X,Y,V, 50, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('S')
ax.set_ylabel('t')
ax.set_zlabel('option price')
plt.show()

# # Plot Delta
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.contour3D(X,Y,V/S, 50, cmap=cm.cool)
# ax.set_xlabel('S')
# ax.set_ylabel('t')
# ax.set_zlabel('Delta')
# plt.show()