import math
import random
import numpy as np
from scipy.stats import norm
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import seaborn as sns
sns.set(color_codes=True)

# Black-Scholes formulas
def option_value_bs(St, K, T, sigma, r, t=0):
    d1 = (math.log(St/K) + (r + sigma ** 2 * 0.5) * (T - t)) / (sigma * math.sqrt(T - t))
    d2 = d1 - sigma * math.sqrt(T - t)
    return St * norm.cdf(d1) - K * math.exp(-r * (T - t)) * norm.cdf(d2)

def Europe_Call_FD(S0=100, K=110, T=1, sigma=0.3, r=0.04, Exp1=-5, Exp2=7, M=1000, N=1000, scheme='FTCS'):
    # Initialization
    X0 = math.log(S0)   
    V = np.zeros((M+2, N+1))
    x_list = np.linspace(Exp1, Exp2, M+2)
    t_list = np.arange(1, N+1)

    dx = x_list[2] - x_list[1]
    dt = T / float(N)

    # Setup boundaries on first column and last row
    V[:, 0] = np.maximum(np.exp(x_list) - K, 0)
    V[-1, 1:] = (np.exp(Exp2) - K) * np.exp(-r * t_list * dt)
    
    # Compute coefficients and construct matrices A and B
    if scheme == 'FTCS':
        alpha =  - (( r - 0.5* sigma**2) * dt / (2 * dx)) + (sigma**2   * dt / (2 * dx**2))
        beta = 1 - (r * dt + sigma**2 * dt / (dx**2))
        gamma = (( r - 0.5* sigma**2) * dt / (2 * dx)) + (sigma**2   * dt / (2 * dx**2))
        
        A = np.diag([alpha for i in range(M-1)], -1) + \
            np.diag([beta for i in range(M)]) + \
            np.diag([gamma for i in range(M-1)], 1)
        
        B = np.identity(M)
        B_inv = B
        for t in t_list:
            V_prev = V[1:-1, t-1]
            k_new = np.zeros(M)
            k_prev = np.zeros(M)
            k_prev[0] = 0
            k_prev[-1] = gamma * V[-1, t-1]

            # Compute V^{n+1}
            V[1:-1, t] = B_inv.dot(A.dot(V_prev) + k_prev - k_new)

    elif scheme == "CN":
        alpha = ((( r - 0.5* sigma**2) * dt / (2 * dx)) - (sigma**2   * dt / (2 * dx**2))) / 2
        beta = (r * dt + sigma**2 * dt / (dx**2)) / 2
        gamma = (- (( r - 0.5* sigma**2) * dt / (2 * dx)) - (sigma**2   * dt / (2 * dx**2))) / 2
        
        A = np.diag([-alpha for i in range(M-1)], -1) + \
            np.diag([1-beta for i in range(M)]) + \
            np.diag([-gamma for i in range(M-1)], 1)
        
        B = np.diag([alpha for i in range(M-1)], -1) + \
            np.diag([1+beta for i in range(M)]) + \
            np.diag([gamma for i in range(M-1)], 1)
        B_inv = np.linalg.inv(B)
    
        # Traverse the V
        for t in t_list:
            V_prev = V[1:-1, t-1]
            k_new = np.zeros(M)
            k_new[0] = 0
            k_new[-1] = gamma * V[-1, t]
            k_prev = np.zeros(M)
            k_prev[0] = 0
            k_prev[-1] = -gamma * V[-1, t-1]
            
            # Compute V^{n+1}
            V[1:-1, t] = B_inv.dot(A.dot(V_prev) + k_prev - k_new)
    
    # Get option value
    option_value = np.interp(X0, x_list, V[:, -1])
    
    return option_value, V, np.exp(x_list), dx, dt


if __name__ == "__main__":

    Exp_low = -6
    Exp_high = 8
    M = 1000
    N = 1000

    """
    FTCS Scheme
    """

    # r = 4%; vol = 30%; S0 = 100; K = 110;
    bs_opt_value = option_value_bs(100, 110, 1, 0.3, 0.04)
    print("Black scholes value:",bs_opt_value)

    opt_val, V, _, dX, dT = Europe_Call_FD(S0=100, K=110, T=1, sigma=0.3, r=0.04, Exp1=Exp_low, Exp2=Exp_high, M=M, N=N, scheme='FTCS')
    print('FTCS option value: %.4f   \nDelta_X: %.4f   \nDelta_tau: %.4f' % (opt_val, dX, dT))
    error = (opt_val / bs_opt_value - 1) * 100
    print('Relative error: %.4f%%' % error)

    # r = 4%; vol = 30%; S0 = 110; K = 110
    bs_opt_value = option_value_bs(110, 110, 1, 0.3, 0.04)
    print("Black scholes value:",bs_opt_value)

    opt_val, V, _, dX, dT = Europe_Call_FD(S0=110, K=110, T=1, sigma=0.3, r=0.04, Exp1=Exp_low, Exp2=Exp_high, M=M, N=N, scheme='FTCS')
    print('FTCS option value: %.4f   \nDelta_X: %.4f   \nDelta_tau: %.4f' % (opt_val, dX, dT))
    error = (opt_val / bs_opt_value - 1) * 100
    print('Relative error: %.4f%%' % error)

    # r = 4%; vol = 30%; S0 = 120; K = 110
    bs_opt_value = option_value_bs(120, 110, 1, 0.3, 0.04)
    print("Black scholes value:",bs_opt_value)

    opt_val, V, _, dX, dT = Europe_Call_FD(S0=120, K=110, T=1, sigma=0.3, r=0.04, Exp1=Exp_low, Exp2=Exp_high, M=M, N=N, scheme='FTCS')
    print('FTCS option value: %.4f   \nDelta_X: %.4f   \nDelta_tau: %.4f' % (opt_val, dX, dT))
    error = (opt_val / bs_opt_value - 1) * 100
    print('Relative error: %.4f%%' % error)

    """
    Crank Nicholsen Scheme
    """

    # r = 4%; vol = 30%; S0 = 100; K = 110;
    bs_opt_value = option_value_bs(100, 110, 1, 0.3, 0.04)
    print("Black scholes value:",bs_opt_value)

    opt_val, V, _, dX, dT = Europe_Call_FD(S0=100, K=110, T=1, sigma=0.3, r=0.04, Exp1=Exp_low, Exp2=Exp_high, M=M, N=N, scheme='CN')
    print('CN option value: %.4f \nDelta_X: %.4f \nDelta_tau: %.4f' % (opt_val, dX, dT))
    error = (opt_val / bs_opt_value - 1) * 100
    print('Relative error: %.4f%%' % error)

    # r = 4%; vol = 30%; S0 = 110; K = 110
    bs_opt_value = option_value_bs(110, 110, 1, 0.3, 0.04)
    print("Black scholes value:",bs_opt_value)

    opt_val, V, _, dX, dT = Europe_Call_FD(S0=110, K=110, T=1, sigma=0.3, r=0.04, Exp1=Exp_low, Exp2=Exp_high, M=M, N=N, scheme='CN')
    print('CN option value: %.4f \nDelta_X: %.4f \nDelta_tau: %.4f' % (opt_val, dX, dT))
    error = (opt_val / bs_opt_value - 1) * 100
    print('Relative error: %.4f%%' % error)

    # r = 4%; vol = 30%; S0 = 120; K = 110
    bs_opt_value = option_value_bs(120, 110, 1, 0.3, 0.04)
    print("Black scholes value:",bs_opt_value)

    opt_val, V, _, dX, dT = Europe_Call_FD(S0=120, K=110, T=1, sigma=0.3, r=0.04, Exp1=Exp_low, Exp2=Exp_high, M=M, N=N, scheme='CN')
    print('CN option value: %.4f \nDelta_X: %.4f \nDelta_tau: %.4f' % (opt_val, dX, dT))
    error = (opt_val / bs_opt_value - 1) * 100
    print('Relative error: %.4f%%' % error)