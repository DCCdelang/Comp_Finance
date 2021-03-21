
import math
import random
import numpy as np
from scipy.stats import norm
from tqdm import tqdm as _tqdm
import copy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

import seaborn as sns
sns.set(color_codes=True)

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)



# Functions for Black-Scholes formulas
def option_value_bs(St, K, T, sigma, r, t=0):
    d1 = (math.log(St/K) + (r + sigma ** 2 * 0.5) * (T - t)) / (sigma * math.sqrt(T - t))
    d2 = d1 - sigma * math.sqrt(T - t)
    
    return St * norm.cdf(d1) - K * math.exp(-r * (T - t)) * norm.cdf(d2)

def hedge_parameter_bs(St, K, T, sigma, r, t=0):
    return norm.cdf((math.log(St/K) + (r + sigma ** 2 * 0.5) * (T - t)) / (sigma * math.sqrt(T - t)))

def FD_Schemes(S0=100, K=110, T=1, v=0.3, r=0.04, M1=-5, M2=7, N_X=1000, N_T=1000, scheme='FTCS'):
    assert scheme in ['FTCS', 'CN'], 'Not a valid scheme type'
    # Initialization
    X0 = math.log(S0)
    
    grid = np.zeros((N_X+2, N_T+1))
    x_values = np.linspace(M1, M2, N_X+2)
    
    delta_x = x_values[2] - x_values[1]
    delta_tau = T / float(N_T)
    
    x_index = np.arange(1, N_X+1)
    tau_index = np.arange(1, N_T+1)
    
    # Setup boundaries on first column and last row
    grid[:, 0] = np.maximum(np.exp(x_values) - K, 0)
    grid[-1, 1:] = (np.exp(M2) - K) * np.exp(-r * tau_index * delta_tau)
    
    # Define coefficients to use in both schemes
    a = (2 * r - v**2) * delta_tau / (4 * delta_x)
    b = v**2 * delta_tau / (2 * delta_x**2)
    c = r * delta_tau + v**2 * delta_tau / (delta_x**2)
    
    # Compute coefficients and construct matrices A and B
    if scheme is 'FTCS':
        alpha =  - a + b
        beta = 1 - c
        gamma = a + b
        
        A = np.diag([alpha for i in range(N_X-1)], -1) + \
            np.diag([beta for i in range(N_X)]) + \
            np.diag([gamma for i in range(N_X-1)], 1)
        
        B = np.identity(N_X)
        B_inv = B
    else:
        alpha = (a - b) / 2
        beta = c / 2
        gamma = (- a - b) / 2
        
        A = np.diag([-alpha for i in range(N_X-1)], -1) + \
            np.diag([1-beta for i in range(N_X)]) + \
            np.diag([-gamma for i in range(N_X-1)], 1)
        
        B = np.diag([alpha for i in range(N_X-1)], -1) + \
            np.diag([1+beta for i in range(N_X)]) + \
            np.diag([gamma for i in range(N_X-1)], 1)
        B_inv = np.linalg.inv(B)
    
    # Traverse the grid
    for t in tau_index:
        # Get V^n from grid
        V_prev = grid[1:-1, t-1]
        
        if scheme is 'FTCS':
            # k_1 is 0
            k_new = np.zeros(N_X)
            
            # Construct k_2
            k_prev = np.zeros(N_X)
            k_prev[0] = 0
            k_prev[-1] = gamma * grid[-1, t-1]
        else:
            # Construct k_1
            k_new = np.zeros(N_X)
            k_new[0] = 0
            k_new[-1] = gamma * grid[-1, t]
            
            # Construct k_2
            k_prev = np.zeros(N_X)
            k_prev[0] = 0
            k_prev[-1] = -gamma * grid[-1, t-1]
        
        # Compute V^{n+1}
        grid[1:-1, t] = B_inv.dot(A.dot(V_prev) + k_prev - k_new)
    
    # Interpolate value of option by looking at the last column of the grid
    opt_value = np.interp(X0, x_values, grid[:, -1])
    print(tau_index.shape, grid.shape, x_index.shape,)
    return opt_value, grid, np.exp(x_values), delta_x, delta_tau

f_M1 = -5
f_M2 = 7
f_N_X = 1000
f_N_T = 1000
print("\n",FD_Schemes(S0=110, K=110, T=1, v=0.3, r=0.04, M1=-5, M2=7, N_X=1000, N_T=1000, scheme='FTCS')[0],"\n")
raise ValueError()


bs_opt_value = option_value_bs(100, 110, 1, 0.3, 0.04)

opt_val, grid, _, dX, dT = FD_Schemes(S0=100, K=110, T=1, v=0.3, r=0.04,
                                      M1=f_M1, M2=f_M2, N_X=f_N_X, N_T=f_N_T, scheme='FTCS')
print('Option value: %.4f     Delta_X: %.4f     Delta_tau: %.4f' % (opt_val, dX, dT))
error = (opt_val / bs_opt_value - 1) * 100
print('Relative error: %.4f%%' % error)

opt_val, grid, _, dX, dT = FD_Schemes(S0=100, K=110, T=1, v=0.3, r=0.04,
                                      M1=f_M1, M2=f_M2, N_X=f_N_X, N_T=f_N_T, scheme='CN')
print('Option value: %.4f     Delta_X: %.4f     Delta_tau: %.4f' % (opt_val, dX, dT))
error = (opt_val / bs_opt_value - 1) * 100
print('Relative error: %.4f%%' % error)

bs_opt_value_2 = option_value_bs(110, 110, 1, 0.3, 0.04)


opt_val, grid, _, dX, dT = FD_Schemes(S0=110, K=110, T=1, v=0.3, r=0.04,
                                      M1=f_M1, M2=f_M2, N_X=f_N_X, N_T=f_N_T, scheme='FTCS')
print('Option value: %.4f     Delta_X: %.4f     Delta_tau: %.4f' % (opt_val, dX, dT))
error = (opt_val / bs_opt_value_2 - 1) * 100
print('Relative error: %.4f%%' % error)