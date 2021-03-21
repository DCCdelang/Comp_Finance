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
    if scheme == 'FTCS':
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
        
        if scheme == 'FTCS':
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
    #print(tau_index.shape, grid.shape, x_index.shape,)
    return opt_value, grid, np.exp(x_values), delta_x, delta_tau

f_M1 = -5
f_M2 = 7
f_N_X = 1000
f_N_T = 1000
#print("\n FD",FD_Schemes(S0=100, K=110, T=1, v=0.3, r=0.04, M1=-5, M2=7, N_X=1000, N_T=1000, scheme='FTCS')[0],"\n")


bs_opt_value = option_value_bs(100, 110, 1, 0.3, 0.04)
#print("BS:",bs_opt_value)
N_X_list = np.linspace(150, 2000, 10)
#N__list = int(np.linspace(150, 2000, 10))
N__list = [150, 300, 500, 750, 1000,  2000]
N__list = range(100, 1500, 25)
#N__list = range(400, 1500, 25)
errorFD = []
errorCN = []
BS = []
# Europe_Call_FD(S0=100, K=110, T=1, sigma=0.3, r=0.04, Exp1=-5, Exp2=7, M=1000, N=1000, scheme='FTCS'):

for N__ in N__list:
    #FD_X = FD_Schemes(S0=100, K=110, T=1, v=0.3, r=0.04, M1=-6, M2=8, N_X=N__, N_T=1000, scheme='FTCS')[0]
    #FD_X = FD_Schemes(S0=100, K=110, T=1, v=0.3, r=0.04, M1=-6, M2=8, N_X=1000, N_T=N__, scheme='FTCS')[0]
    FD_X = FD_Schemes(S0=100, K=110, T=1, v=0.3, r=0.04, M1=-6, M2=8, N_X=N__, N_T=N__, scheme='FTCS')[0]
    #print(FD_X)
    #CN_X = FD_Schemes(S0=100, K=110, T=1, v=0.3, r=0.04, M1=-6, M2=8, N_X=N__, N_T=1000, scheme='CN')[0]
    #CN_X = FD_Schemes(S0=100, K=110, T=1, v=0.3, r=0.04, M1=-6, M2=8, N_X=1000, N_T=N__, scheme='CN')[0]
    CN_X = FD_Schemes(S0=100, K=110, T=1, v=0.3, r=0.04, M1=-6, M2=8, N_X=N__, N_T=N__, scheme='CN')[0]

    #BS.append(option_value_bs(100, 110, 1, 0.3, 0.04))
    BS.append(0)
    errorFD.append(np.abs(FD_X-option_value_bs(100, 110, 1, 0.3, 0.04)))
    errorCN.append(np.abs(CN_X-option_value_bs(100, 110, 1, 0.3, 0.04)))
    #errorFD.append(100*(1-FD_X/option_value_bs(100, 110, 1, 0.3, 0.04)))
    #errorCN.append(100*(1-CN_X/option_value_bs(100, 110, 1, 0.3, 0.04)))

plt.plot(N__list, errorFD, label="FTCS")
plt.plot(N__list, errorCN, label="CN")
plt.plot(N__list, BS, "--r")
plt.xlabel("N, M", size = 14)
plt.ylabel("error", size = 14)
#plt.yscale("log")
#plt.xscale("log")
plt.legend()
plt.show()
"""
M_list = np.array([50, 100, 200, 400, 800, 1600])
#N_list = np.array([100, 200, 400, 800, 1600, 3200])
N_list = np.array([50, 100, 200, 400, 800, 1600])
errorFD = []
errorCN = []

for M_, N_ in zip(M_list, N_list):
    FD_X = FD_Schemes(S0=100, K=110, T=1, v=0.3, r=0.04, M1=-6, M2=8, N_X=M_, N_T=N_, scheme='FTCS')[0]
    CN_X = FD_Schemes(S0=100, K=110, T=1, v=0.3, r=0.04, M1=-6, M2=8, N_X=M_, N_T=N_, scheme='CN')[0]
    errorFD.append(np.abs(FD_X-option_value_bs(100, 110, 1, 0.3, 0.04)))
    errorCN.append(np.abs(CN_X-option_value_bs(100, 110, 1, 0.3, 0.04)))

plt.plot(M_list, errorFD, label="FTCS")
plt.plot(M_list, errorCN, label="CN")
#plt.plot(N__list, errorT, label="changing N_T")
#plt.plot(N__list, BS, "--r")
plt.xlabel("M, N", size = 14)
plt.ylabel("error", size = 14)
#plt.yscale("log")
#plt.xscale("log")
plt.legend()
plt.show()
"""



"""
# opt_val, grid, _, dX, dT = FD_Schemes(S0=100, K=110, T=1, v=0.3, r=0.04,
#                                       M1=f_M1, M2=f_M2, N_X=f_N_X, N_T=f_N_T, scheme='FTCS')
# print('Option value: %.4f     Delta_X: %.4f     Delta_tau: %.4f' % (opt_val, dX, dT))
# error = (opt_val / bs_opt_value - 1) * 100
# print('Relative error: %.4f%%' % error)

# opt_val, grid, _, dX, dT = FD_Schemes(S0=100, K=110, T=1, v=0.3, r=0.04,
#                                       M1=f_M1, M2=f_M2, N_X=f_N_X, N_T=f_N_T, scheme='CN')
# print('Option value: %.4f     Delta_X: %.4f     Delta_tau: %.4f' % (opt_val, dX, dT))
# error = (opt_val / bs_opt_value - 1) * 100
# print('Relative error: %.4f%%' % error)

# bs_opt_value_2 = option_value_bs(110, 110, 1, 0.3, 0.04)


# opt_val, grid, _, dX, dT = FD_Schemes(S0=110, K=110, T=1, v=0.3, r=0.04,
#                                       M1=f_M1, M2=f_M2, N_X=f_N_X, N_T=f_N_T, scheme='FTCS')
# print('Option value: %.4f     Delta_X: %.4f     Delta_tau: %.4f' % (opt_val, dX, dT))
# error = (opt_val / bs_opt_value_2 - 1) * 100
# print('Relative error: %.4f%%' % error)

def plot_3d_grid(S0, K, T, v, r, M1, M2, N_X, N_T, scheme, restrict=True, savefig=False):
    print(f'Scheme {scheme}')
    opt_value, grid, _, _, _ = FD_Schemes(S0=S0, K=K, T=T, v=v, r=r,
                                          M1=M1, M2=M2, N_X=N_X, N_T=N_T, scheme=scheme)
    
    fig = plt.figure(figsize=(10, 8))
    ax = Axes3D(fig)
    if restrict:
        x_range = np.linspace(M1, M2, f_N_X+2)
        idx = np.where(x_range > 3.5)[0][0]
        idx2 = np.where(x_range > 5)[0][0]
        print(idx,idx2)
        grid = grid[idx:idx2, :]
    else:
        idx = 0
    S_list = []
    for value in np.linspace(M1,M2, N_X+2)[idx:idx2]:
        S_list.append(np.exp(value))
    t, S = np.meshgrid(np.linspace(0, 1, N_T+1), S_list)
    
    ax.plot_surface(S, t, grid, cmap='coolwarm', linewidth=0, antialiased=True)
    ax.view_init(20, 100)
    ax.set_xlabel('S')
    ax.set_ylabel('tau')
    ax.set_zlabel('option price')
    if savefig:
        plt.savefig(f'3d_grid_{scheme}_restrict_{restrict}.png', dpi=200)
    plt.show()



f_M1 = -5
f_M2 = 7
f_N_X = 1000
f_N_T = 1000

plot_3d_grid(S0=100, K=110, T=1, v=0.3, r=0.04, M1=f_M1, M2=f_M2, N_X=f_N_X, N_T=f_N_T, scheme='CN', restrict=True)






def compute_and_plot_delta(S0=100, K=110, T=1, v=0.3, r=0.04, M1=-5, M2=7, N_X=2000, N_T=1000, scheme='CN',
                           restrict=True, S_min=10, S_max=100, savefig=False):
    # Get FD Scheme data
    _, grid, S0_val, _, _ = FD_Schemes(S0, K, T, v, r, M1, M2, N_X, N_T, scheme=scheme)
    
    x_values = np.linspace(M1, M2, N_X+2)
    V = grid[:, -1]
    
    # Compute deltas for FD Schemes, either restricted to interval or on full domain
    fd_delta = []
    bs_delta = []
    if not restrict:
        for i in np.arange(1, N_X+1):
            fd_delta.append((V[i+1] - V[i-1]) / (S0_val[i+1] - S0_val[i-1]))
            bs_delta.append(hedge_parameter_bs(S0_val[i], K, T, v, r))

        S0_val = S0_val[1:-1]
    else:
        S_values = np.arange(S_min-1, S_max+1, 1)
        
        for i in np.arange(1, len(S_values)-1):
            V_i = np.interp(np.log(S_values[i-1]), x_values, V)
            V_i_next = np.interp(np.log(S_values[i+1]), x_values, V)
            fd_delta.append((V_i_next - V_i) / (S_values[i+1] - S_values[i-1]))
            bs_delta.append(hedge_parameter_bs(S_values[i], K, T, v, r))
        S0_val = S_values[1:-1]
    
    # Plot lines
    plt.figure(figsize=(12, 10))
    plt.plot(S0_val, fd_delta, label=scheme, alpha=0.8)
    plt.plot(S0_val, bs_delta, label='Black-Scholes', alpha=0.6)
    plt.xlabel('$S_0$')
    plt.ylabel('Delta')
    plt.legend()
    if savefig:
        plt.savefig(f'delta_N_X_{N_X}_restrict_{restrict}.png', dpi=200)
    plt.show()
    
    if restrict:
        fd_d = np.array(fd_delta)
        bs_d = np.array(bs_delta)
        print(f'Mean absolute error: {np.mean(np.abs(fd_d - bs_d)):.6f}')

compute_and_plot_delta(N_X=2000, N_T=1000, scheme='CN', restrict=False)
"""