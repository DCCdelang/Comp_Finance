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

from Finite_EU_Call import Europe_Call_FD


M_list = np.array([50, 100, 200, 400, 800, 1600])
#N_list = np.array([100, 200, 400, 800, 1600, 3200])
N_list = np.array([50, 100, 200, 400, 800, 1600])
errorFD = []
errorCN = []

for M_, N_ in zip(M_list, N_list):
    FD_X = Europe_Call_FD(S0=100, K=110, T=1, v=0.3, r=0.04, M1=-6, M2=8, N_X=M_, N_T=N_, scheme='FTCS')[0]
    CN_X = Europe_Call_FD(S0=100, K=110, T=1, v=0.3, r=0.04, M1=-6, M2=8, N_X=M_, N_T=N_, scheme='CN')[0]
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



