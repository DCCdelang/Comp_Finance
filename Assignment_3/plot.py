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
    S,t = np.meshgrid(np.linspace(0, 1, N+1), S_list)


    ax.plot_surface(t, S, V, cmap='Greens', linewidth=0, antialiased=True)
    ax.view_init(20, 100)
    ax.tick_params(labelsize=16)
    ax.set_xlabel('S0', fontsize=16)
    ax.set_ylabel('t', fontsize=16)
    ax.set_zlabel('option price', fontsize=16)
    if savefig:
        plt.savefig(f'3d_grid_{scheme}_restrict_{restrict}.pdf', dpi=200)
    plt.show()

def hedge_parameter_bs(St, K, T, sigma, r, t=0):
    return norm.cdf((math.log(St/K) + (r + sigma ** 2 * 0.5) * (T - t)) / (sigma * math.sqrt(T - t)))

def compute_and_plot_delta(S0=100, K=110, T=1, sigma=0.3, r=0.04, Exp1=-5, Exp2=7, M=2000, N=1000, scheme='CN',restrict=True, S_min=10, S_max=200, savefig=True, error=False):
    _, grid, S0_val, _, _ = Europe_Call_FD(S0, K, T, sigma, r, Exp1, Exp2, M, N, scheme=scheme)
    
    x_values = np.linspace(Exp1, Exp2, M+2)
    V = grid[:, -1]
    
    # Compute deltas for FD Schemes, either restricted to interval or on full domain
    fd_delta = []
    bs_delta = []

    S_values = np.arange(S_min-1, 140+1, 1)
    
    for i in np.arange(1, len(S_values)-1):
        V_i = np.interp(np.log(S_values[i-1]), x_values, V)
        V_i_next = np.interp(np.log(S_values[i+1]), x_values, V)
        fd_delta.append((V_i_next - V_i) / (S_values[i+1] - S_values[i-1]))
        bs_delta.append(hedge_parameter_bs(S_values[i], K, T, sigma, r))
    S0_val = S_values[1:-1]
    
    if error == True:
        error = []
        for i in range(len(bs_delta)):
            error.append(abs(bs_delta[i] - fd_delta[i]))

        plt.plot(S0_val, error, label=scheme, alpha=0.5)
        plt.show()
    

    # Plot lines
    plt.plot(S0_val, fd_delta, label=scheme, alpha=0.5)

    if scheme != "CN":
        plt.plot(S0_val, bs_delta,"--", label='Black-Scholes')

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('S0', fontsize=14)
    plt.ylabel('Delta', fontsize=14)
    plt.legend()
    if savefig:
        plt.savefig(f'delta.pdf', dpi=200)
    


if __name__ == "__main__":

    """ Delta plot """
    # compute_and_plot_delta(M=2000, N=1000, scheme='CN', restrict=True, savefig=False)

    """ 3D plot """
    plot_3D(S0=100, K=110, T=1, sigma=0.3, r=0.04, Exp1=Exp_low, Exp2=Exp_high, M=M, N=N, scheme='FTCS', restrict=True)

    # plot_3D(S0=110, K=110, T=1, sigma=0.3, r=0.04, Exp1=Exp_low, Exp2=Exp_high, M=M, N=N, scheme='CN', restrict=True)

    """ 2D plot"""
    # plot_2D(S0=100, K=110, T=1, sigma=0.3, r=0.04, Exp1=Exp_low, Exp2=Exp_high, M=M, N=N, scheme='FTCS', restrict=True)
