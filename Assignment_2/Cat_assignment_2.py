import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import *
from scipy.stats import norm, gmean
import seaborn as sns
import random

#%%

def option_price(K, S, r, sigma, T, pay_offs):
    return np.exp(-r * T)*(np.mean(pay_offs))

def ST(K, S, r, sigma, T):
    # np.random.seed(42)
    Z = np.random.normal()
    ST = S* (np.exp( (r-0.5*sigma**2)*T + sigma*np.sqrt(T)*Z))
    
    return max(K-ST, 0)

def N_(x):
    #'Cumulative distribution function for the standard normal distribution'

    # print((1.0 + erf(x / sqrt(2.0))) / 2.0)
    
    return norm.cdf(x)

def black_scholes_c(S,N,T,sigma,r,K):
    d = ((np.log(S/K) + ((r-(sigma**2)/2)) * T )/(sigma*np.sqrt(T)))
    d_1 = d + sigma* np.sqrt(T)
    return (S*N_(d_1)) - (K*np.exp(-r*T)*N_(d))

def black_scholes_p(S,N,T,sigma,r,K):
    d = ((np.log(S/K) + ((r-(sigma**2)/2)) * T )/(sigma*np.sqrt(T)))
    d_1 = d + sigma* np.sqrt(T)
    return  (K*np.exp(-r*T)*N_(-d)) - (S*N_(-d_1))

def asian_anal(S,N,T,sigma,r,K):
    sigma_tilde = sigma * sqrt((2*N+1)/(6*(N+1)))
    r_tilde = ((r-sigma**2/2)+sigma_tilde**2)/2

    d_tilde_1 = ((np.log(S/K) + ((r_tilde+(sigma_tilde**2)/2)) * T )/(sigma_tilde*np.sqrt(T)))
    d_tilde_2 = ((np.log(S/K) + ((r_tilde-(sigma_tilde**2)/2)) * T )/(sigma_tilde*np.sqrt(T)))

    return np.exp(-r*T)*(S*np.exp(r_tilde*T)*N_(d_tilde_1)-K*N_(d_tilde_2))

def asian_MC(S,N,T,sigma,r,K, n, type_op, param):
    payoff = []
    payoff_a = []
    payoff_g = []
    sim = []
    T_i = T/N
    #data = {"Values":payoff, "Simulation":sim}
    #df = pd.DataFrame() 
    for i in range(n):
        S_ti_ar = []
        S_ti_geo = []
        ST = S

        if type_op == "arithmetic":
            for j in range(int(N)):
                Z = np.random.normal()
                ST = ST * (np.exp( (r-0.5*sigma**2)*T_i + sigma*np.sqrt(T_i)*Z))
                S_ti_ar.append(ST)
        
            payoff.append(max(np.mean(S_ti_ar)-K, 0))
            sim.append(n)

        elif type_op == "geometric":
            for j in range(int(N)):
                Z = np.random.normal()
                ST = ST * np.exp( (r-0.5*sigma**2)*T_i + sigma*np.sqrt(T_i)*Z)
                S_ti_geo.append(ST)


            payoff.append(max(gmean(S_ti_geo)-K, 0))
            sim.append(n)

        elif type_op == "control":
            for j in range(int(N)):
                Z = np.random.normal()
                
                ST = ST * (np.exp( (r-0.5*sigma**2)*T_i + sigma*np.sqrt(T_i)*Z))
                S_ti_geo.append(ST)
                S_ti_ar.append(ST)

            payoff_g.append(max(gmean(S_ti_geo)-K, 0))            
            payoff_a.append(max(np.mean(S_ti_ar)-K, 0))            
            sim.append(n)



    if type_op == "control":
        K_ = np.repeat(K,n)
        sigma_= np.repeat(sigma,n)
        N_= np.repeat(N,n)
        n_ = np.repeat(n, n)

        control = np.exp(-r * T) * (np.mean(payoff_a)-np.mean(payoff_g))+asian_anal(S,N,T,sigma,r,K)
        var_contr = np.exp(-r * T)**2 * ( np.var(payoff_a) + np.var(payoff_g) - 2*(np.cov(payoff_a, payoff_g)[0,1]))
        var_MC = np.exp(-r * T)**2 * np.var(payoff_a)
        #print(asian_anal(S,N,T,sigma,r,K)-np.asarray(payoff_g))
        data = {"Arithmetic":np.exp(-r * T) * np.asarray(payoff_a),
        "Geometric":np.exp(-r * T) * np.asarray(payoff_g),
        "Control": np.exp(-r * T) * (np.asarray(payoff_a)+asian_anal(S,N,T,sigma,r,K)-np.asarray(payoff_g)),
        "K":K_,
        "sigma": sigma_,
        "N": N_,
        "n":n_,}
        df = pd.DataFrame(data)
        if param == "K":
            df.to_csv(f"K/asian_MC_{n}_{K}_{sigma}_{N}.csv")
        elif param == "N":
            df.to_csv(f"N/asian_MC_{n}_{K}_{sigma}_{N}.csv")
        elif param == "sigma":
            df.to_csv(f"sigma/asian_MC_{n}_{K}_{sigma}_{N}.csv")
        elif param == "n":
            df.to_csv(f"path/asian_MC_{n}_{K}_{sigma}_{N}.csv")
        elif param == "none":
            df.to_csv(f"asian_MC_{n}_{K}_{sigma}_{N}.csv")
        
        #print(np.mean(asian_anal(S,N,T,sigma,r,K)-np.asarray(payoff_g)))
        return control, np.exp(-r * T) * (np.mean(payoff_a)), var_contr, var_MC

    option_values = np.exp(-r * T) * np.asarray(payoff)
    #print(option_values)
    data = {"Values": option_values, "Simulation":sim}
    df = pd.DataFrame(data) 
    df.to_csv(f"asian_MC_{n}_{K}_{sigma}_{N}.csv")
    return np.exp(-r * T) * np.mean(payoff), np.std(payoff)/sqrt(n)

#%%
#### ASIAN OPTION
K = 99
S = 100
r = 0.06
sigma = 0.2
T = 1
N = 365
M = n = 1000

asian_analytical = asian_anal(S,N,T,sigma,r,K)
print(asian_analytical)
asian_geom = asian_MC(S,N,T,sigma,r,K, n, type_op = "geometric", param="None")
print(asian_geom)


#%%
nn = [100,500, 1000,5000, 10000, 50000, 100000, 500000]
#nn = [100000]
asian_MC_list = []
asian_anal_list = []
standard_error = []
columns = ["Values", "Simulation"]
df_final = pd.DataFrame()
for n in nn:
    MC = asian_MC(S,N,T,sigma,r,K,n,type_op = "geometric", param="None")
    #asian_MC_list.append(MC[0])
    #standard_error.append(MC[1])
    asian_anal_list.append(asian_anal(S,N,T,sigma,r,K))

#%%
nn = [100,500, 1000,5000, 10000, 50000, 100000, 500000]
frames = [ pd.read_csv(f"asian_MC_{n}.csv") for n in nn ]
result = pd.concat(frames)
print(result)
result.to_csv("asian_MC_final")
#%%
"""
3.1: plot for comparing analytical and MC values
"""

#df = pd.read_csv("jToverN/asian_MC_final")
sns.lineplot(data=result, x="Simulation", y="Values", label = "Monte Carlo")
plt.plot(nn, asian_anal_list, label = "Analytical")
plt.xscale("log")
plt.legend()
plt.savefig("Asian_1_2.pdf")
plt.show()
#df_final.append(df, ignore_index=True)
#df_final.to_csv(f"asian_MC_final.csv")
"""
plt.plot(nn, asian_MC_list, label = "Monte Carlo")
plt.plot(nn, asian_anal_list, label = "Analytical")
plt.plot(nn, np.asarray(asian_MC_list) + np.asarray(standard_error))
plt.plot(nn, np.asarray(asian_MC_list) - np.asarray(standard_error))
plt.legend()
plt.show()
"""
#plt.xscale("log") 
#plt.savefig("Convergence_option_price.pdf")


#%%
"""
3.3.a: Apply the control variates technique for the calculation of 
the value of the Asian option based on arithmetic averages.
"""
K = 99
S = 100
r = 0.06
sigma = 0.2
T = 1
N = 365
n = 1000

MC = asian_MC(S,N,T,sigma,r,K, n, type_op="control", param="None")
contr = MC[0]
asian_mont = MC[1]

print(contr, asian_mont)


#%%
"""
3.3.b: different parameter settings.
"""
S = 100
r = 0.06
sigma = 0.2
T = 1
N = 365
n = 10000

# strike
control = []
asian_mont = []
var_c = []
var_MC = []
k = np.linspace(50, 99, 99-50+1)
for K in k:
    MC = asian_MC(S,N,T,sigma,r,K, n, type_op = "control", param="K")
    control.append(MC[0])
    asian_mont.append(MC[1])
    var_c.append(MC[2])
    var_MC.append(MC[3])

frames = [ pd.read_csv(f"K/asian_MC_{n}_{K}_{sigma}_{N}.csv") for K in k ]
result = pd.concat(frames)
print(result)
result.to_csv("asian_MC_final_K")

data = {"K": k, "var_c":var_c, "var_MC": var_MC}
df = pd.DataFrame(data) 
df.to_csv(f"K/asian_var_K.csv")

sns.lineplot(data=result, x="K", y="Arithmetic", label = "MC")
sns.lineplot(data=result, x="K", y="Control", label = "Control variate")
plt.ylabel("Option value")
plt.legend()
plt.savefig("Asian_3_31.pdf")
plt.show()

plt.plot(k, var_MC, label="MC")
plt.plot(k, var_c, label="Control")
plt.xlabel("K", fontsize=14)
plt.ylabel("Variance", fontsize=14)
plt.legend()
plt.savefig("Asian_3_31_var.pdf")
plt.show()

#%%
# number of timesteps
K = 99
S = 100
r = 0.06
sigma = 0.2
T = 1
n = 10000
#NN = np.linspace(2, 365*2, 365*2-2+1)
NN = [365, 52, 24, 12, 6]
control = []
asian_mont = []
var_c = []
var_MC = []

for N in NN:
    MC = asian_MC(S,N,T,sigma,r,K, n, type_op = "control", param="N")
    control.append(MC[0])
    asian_mont.append(MC[1])
    var_c.append(MC[2])
    var_MC.append(MC[3])

frames = [ pd.read_csv(f"N/asian_MC_{n}_{K}_{sigma}_{N}.csv") for N in NN ]
result = pd.concat(frames)
print(result)
result.to_csv("N/asian_MC_final_N")

data = {"N": NN, "var_c":var_c, "var_MC": var_MC}
df = pd.DataFrame(data) 
df.to_csv(f"N/asian_var_N.csv")
    
sns.lineplot(data=result, x="N", y="Arithmetic", label = "MC")
sns.lineplot(data=result, x="N", y="Control", label = "Control variate")
plt.ylabel("Option value")
plt.legend()
plt.savefig("Asian_3_32.pdf")
plt.show()

plt.plot(NN, var_MC, label="MC")
plt.plot(NN, var_c, label="Control")
plt.xlabel("N", fontsize=14)
plt.ylabel("Variance", fontsize=14)
plt.legend()
plt.savefig("Asian_3_32_var.pdf")
plt.show()
#%%
# sigma
K = 99
S = 100
r = 0.06
T = 1
N = 365
n = 10000
vol = np.linspace(0.1, 1, 10)
control = []
asian_mont = []
var_c = []
var_MC = []

for sigma in vol:
    MC = asian_MC(S,N,T,sigma,r,K, n, type_op = "control", param="sigma")
    control.append(MC[0])
    asian_mont.append(MC[1])
    var_c.append(MC[2])
    var_MC.append(MC[3])

frames = [ pd.read_csv(f"sigma/asian_MC_{n}_{K}_{sigma}_{N}.csv") for sigma in vol ]
result = pd.concat(frames)
print(result)
result.to_csv("sigma/asian_MC_final_sigma")

data = {"sigma": vol, "var_c":var_c, "var_MC": var_MC}
df = pd.DataFrame(data) 
df.to_csv(f"sigma/asian_var_sigma.csv")

sns.lineplot(data=result, x="sigma", y="Arithmetic", label = "MC")
sns.lineplot(data=result, x="sigma", y="Control", label = "Control variate")
plt.ylabel("Option value")
plt.legend()
plt.savefig("Asian_3_33.pdf")
<<<<<<< HEAD
plt.show()
=======
plt.show()


plt.plot(vol, var_MC, label="MC")
plt.plot(vol, var_c, label="Control")
plt.xlabel("sigma",  fontsize=14)
plt.ylabel("Variance", fontsize=14)
plt.legend()
plt.savefig("Asian_3_33_var.pdf")
plt.show()
#%%
# number of path
K = 99
S = 100
r = 0.06
sigma = 0.2
T = 1
N = 365
n = 100
nn = [100,500, 1000,5000, 10000, 50000, 100000, 500000]
control = []
asian_mont = []
var_c = []
var_MC = []

for n in nn:
    MC = asian_MC(S,N,T,sigma,r,K, n, type_op = "control", param="n")
    control.append(MC[0])
    asian_mont.append(MC[1])
    var_c.append(MC[2])
    var_MC.append(MC[3])

frames = [ pd.read_csv(f"path/asian_MC_{n}_{K}_{sigma}_{N}.csv") for n in nn ]
result = pd.concat(frames)
print(result)
result.to_csv("path/asian_MC_final_n")

data = {"path": nn, "var_c":var_c, "var_MC": var_MC}
df = pd.DataFrame(data) 
df.to_csv(f"path/asian_var_path.csv")

sns.lineplot(data=result, x="n", y="Arithmetic", label = "MC")
sns.lineplot(data=result, x="n", y="Control", label = "Control variate")
plt.ylabel("Option value")
plt.xscale("log")
plt.legend()
plt.savefig("Asian_3_34.pdf")
plt.show()

plt.plot(nn, var_MC, label="MC")
plt.plot(nn, var_c, label="Control")
plt.xscale("log")
plt.xlabel("n", fontsize=14)
plt.ylabel("Variance", fontsize=14)
plt.legend()
plt.savefig("Asian_3_34_var.pdf")
plt.show()
#%%
df = pd.read_csv("path/asian_MC_final_n")
sd_con = []
sd_MC = []
for n in nn:
    sd_con.append(np.std(df.loc[df["n"]==n]["Control"]))
    sd_MC.append(np.std(df.loc[df["n"]==n]["Arithmetic"]))
plt.plot(nn, var_MC, label="MC")
plt.plot(nn, var_c, label="Control")
plt.xscale("log")
plt.xlabel("n", fontsize=14)
plt.ylabel("Variance", fontsize=14)
plt.legend()
#plt.savefig("Asian_3_34_var.pdf")
plt.show()
plt.plot(nn, sd_MC, label="MC")
plt.plot(nn, sd_con, label="Control")
plt.xscale("log")
plt.xlabel("n", fontsize=14)
plt.ylabel("SD", fontsize=14)
plt.legend()
#plt.savefig("Asian_3_34_var.pdf")
plt.show()
#%%
>>>>>>> 4f79de6c5bb7da537011827d2c117560d590780b
