import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

from Finite_EU_Call import Europe_Call_FD

Exp_low = -6
Exp_high = 8
M = 1000
N = 1000

def plot_2D(S0, K, T, sigma, r, Exp1, Exp2, M, N, scheme, restrict=True, savefig=False):
    _, V, _, _, _ = Europe_Call_FD(S0=S0, K=K, T=T, sigma=sigma, r=r, Exp1=Exp1, Exp2=Exp2, M=M, N=N, scheme=scheme)

    if restrict:
        x_range = np.linspace(Exp1, Exp2, M+2)
        idx = np.where(x_range > 3.5)[0][0]
        idx2 = np.where(x_range > 5)[0][0]
        V = V[idx:idx2, :]
    else:
        idx = 0
        idx2 = -1

    S_list = []
    for value in np.linspace(Exp1,Exp2, M+2)[idx:idx2]:
        S_list.append(np.exp(value))

    plt.plot(S_list, V[0])
    plt.show()
    
    t, S = np.meshgrid(np.linspace(0, 1, N+1), S_list)



def plot_3D(S0, K, T, sigma, r, Exp1, Exp2, M, N, scheme, restrict=True, savefig=False):
    _, V, _, _, _ = Europe_Call_FD(S0=S0, K=K, T=T, sigma=sigma, r=r, Exp1=Exp1, Exp2=Exp2, M=M, N=N, scheme=scheme)
    
    fig = plt.figure(figsize=(10, 8))
    ax = Axes3D(fig)
    if restrict:
        x_range = np.linspace(Exp1, Exp2, M+2)
        idx = np.where(x_range > 3.5)[0][0]
        idx2 = np.where(x_range > 5)[0][0]
        V = V[idx:idx2, :]
    else:
        idx = 0
        idx2 = -1
    S_list = []
    for value in np.linspace(Exp1,Exp2, M+2)[idx:idx2]:
        S_list.append(np.exp(value))
    t, S = np.meshgrid(np.linspace(0, 1, N+1), S_list)
    
    ax.plot_surface(S, t, V, cmap='binary', linewidth=0, antialiased=True)
    ax.view_init(20, 100)
    ax.set_xlabel('S')
    ax.set_ylabel('t')
    ax.set_zlabel('option price')
    if savefig:
        plt.savefig(f'3d_grid_{scheme}_restrict_{restrict}.png', dpi=200)
    plt.show()

def hedge_parameter_bs(St, K, T, sigma, r, t=0):
    return norm.cdf((math.log(St/K) + (r + sigma ** 2 * 0.5) * (T - t)) / (sigma * math.sqrt(T - t)))

def compute_and_plot_delta(S0=100, K=110, T=1, sigma=0.3, r=0.04, Exp1=-5, Exp2=7, M=2000, N=1000, scheme='CN',restrict=True, S_min=10, S_max=200, savefig=False):
    _, grid, S0_val, _, _ = Europe_Call_FD(S0, K, T, sigma, r, Exp1, Exp2, M, N, scheme=scheme)
    
    x_values = np.linspace(Exp1, Exp2, M+2)
    V = grid[:, -1]
    
    # Compute deltas for FD Schemes, either restricted to interval or on full domain
    fd_delta = []
    bs_delta = []
    if not restrict:
        for i in np.arange(1, M+1):
            fd_delta.append((V[i+1] - V[i-1]) / (S0_val[i+1] - S0_val[i-1]))
            bs_delta.append(hedge_parameter_bs(S0_val[i], K, T, sigma, r))

        S0_val = S0_val[1:-1]
    else:
        S_values = np.arange(S_min-1, S_max+1, 1)
        
        for i in np.arange(1, len(S_values)-1):
            V_i = np.interp(np.log(S_values[i-1]), x_values, V)
            V_i_next = np.interp(np.log(S_values[i+1]), x_values, V)
            fd_delta.append((V_i_next - V_i) / (S_values[i+1] - S_values[i-1]))
            bs_delta.append(hedge_parameter_bs(S_values[i], K, T, sigma, r))
        S0_val = S_values[1:-1]
    
    # Plot lines
    plt.plot(S0_val, fd_delta, label=scheme, alpha=0.8)
    plt.plot(S0_val, bs_delta, label='Black-Scholes', alpha=0.6)
    plt.xlabel('$S_0$')
    plt.ylabel('Delta')
    plt.legend()
    if savefig:
        plt.savefig(f'delta_N_X_{M}_restrict_{restrict}.png', dpi=200)
    plt.show()
    
    if restrict:
        fd_d = np.array(fd_delta)
        bs_d = np.array(bs_delta)
        print(f'Mean absolute error: {np.mean(np.abs(fd_d - bs_d)):.6f}')

""" Delta plot """
# compute_and_plot_delta(M=2000, N=1000, scheme='CN', restrict=True, savefig=False)

""" 3D plot """
# plot_3D(S0=100, K=110, T=1, sigma=0.3, r=0.04, Exp1=Exp_low, Exp2=Exp_high, M=M, N=N, scheme='FTCS', restrict=True)

# plot_3D(S0=110, K=110, T=1, sigma=0.3, r=0.04, Exp1=Exp_low, Exp2=Exp_high, M=M, N=N, scheme='CN', restrict=True)

""" 2D plot"""
plot_2D(S0=100, K=110, T=1, sigma=0.3, r=0.04, Exp1=Exp_low, Exp2=Exp_high, M=M, N=N, scheme='FTCS', restrict=True)
