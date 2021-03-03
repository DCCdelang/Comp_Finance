#%%
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

<<<<<<< HEAD
def asian_MC(S,N,T,r,K, n, a = "geometric"):
=======
def asian_MC(S,N,T,sigma,r,K, n, type_op, param):
>>>>>>> 032ba89957e7c380e10ab087571f80d57358fab3
    payoff = []
    sim = []
    T_i = T/N
    #data = {"Values":payoff, "Simulation":sim}
    #df = pd.DataFrame() 
    for i in range(n):
<<<<<<< HEAD
        S_ti_ar = 0

        ST=S

        if a == "arithmetic":
            Z = np.random.normal()
            for j in range(N):
                T_i = T/N
                ST = ST * (np.exp( (r-0.5*sigma**2)*T_i + sigma*np.sqrt(T_i)*Z))
=======
        S_ti_ar = []
        S_ti_geo = []
        ST = ST_g = ST_a = S

        if type_op == "arithmetic":
            for j in range(int(N)):
                Z = np.random.normal()
                ST = ST + (np.exp( (r-0.5*sigma**2)*T_i + sigma*np.sqrt(T_i)*Z))
>>>>>>> 032ba89957e7c380e10ab087571f80d57358fab3
                S_ti_ar.append(ST)
        
            payoff.append(max(np.mean(S_ti_ar)-K, 0))
            sim.append(n)

<<<<<<< HEAD
        elif a == "geometric":
            S_ti_geo = [] 
            for j in range(N):
                Z = np.random.normal(0,1)
                T_i = T/N
=======
        elif type_op == "geometric":
            for j in range(int(N)):
                Z = np.random.normal()
>>>>>>> 032ba89957e7c380e10ab087571f80d57358fab3
                ST = ST * np.exp( (r-0.5*sigma**2)*T_i + sigma*np.sqrt(T_i)*Z)
                S_ti_geo.append(ST)

            payoff.append(max(gmean(S_ti_geo)-K, 0))
            sim.append(n)
<<<<<<< HEAD
    
    data = {"Values":payoff, "Simulation":sim}
    df = pd.DataFrame(data) 
    df.to_csv(f"asian_MC_{n}.csv")
    return np.exp(-r * T) * np.mean(payoff), np.std(payoff)/sqrt(N)

=======

        elif type_op == "control":
            for j in range(int(N)):
                Z = np.random.normal()
                
                ST_g = ST_g * (np.exp( (r-0.5*sigma**2)*T_i + sigma*np.sqrt(T_i)*Z))
                S_ti_geo.append(ST_g)
                
                ST_a = ST_a + (np.exp( (r-0.5*sigma**2)*T_i + sigma*np.sqrt(T_i)*Z))
                S_ti_ar.append(ST_a)

            payoff_g.append(max(gmean(S_ti_geo)-K, 0))            
            payoff_a.append(max(np.mean(S_ti_ar)-K, 0))            
            sim.append(n)



    if type_op == "control":
        K_ = np.repeat(K,n)
        sigma_= np.repeat(sigma,n)
        N_= np.repeat(N,n)

        control = np.exp(-r * T) * (np.mean(payoff_a)+asian_anal(S,N,T,sigma,r,K)-np.mean(payoff_g))
        #var_contr = np.exp(-r * T)**2 * ( np.var(payoff_a) + np.var(payoff_g) - 2*np.cov(payoff_a, payoff_g))
        #var_MC = np.var(payoff_a)
        #print(len(np.exp(-r * T) * np.asarray(payoff_a)),
        #len(np.exp(-r * T) * np.asarray(payoff_g)),
        #len(np.exp(-r * T) * (np.asarray(payoff_a)+asian_anal(S,N,T,sigma,r,K)-np.asarray(payoff_g))),
        #len(sim), len(K_), len(sigma_), len(N_))
        data = {"Arithmetic":np.exp(-r * T) * np.asarray(payoff_a),
        "Geometric":np.exp(-r * T) * np.asarray(payoff_g),
        "Control": np.exp(-r * T) * (np.asarray(payoff_a)+asian_anal(S,N,T,sigma,r,K)-np.asarray(payoff_g)),
        "K":K_,
        "sigma": sigma_,
        "N": N_}
        df = pd.DataFrame(data)
        if param == "K":
            df.to_csv(f"K/asian_MC_{n}_{K}_{sigma}_{N}.csv")
        elif param == "N":
            df.to_csv(f"N/asian_MC_{n}_{K}_{sigma}_{N}.csv")
        elif param == "sigma":
            df.to_csv(f"sigma/asian_MC_{n}_{K}_{sigma}_{N}.csv")
        elif param == "none":
            df.to_csv(f"asian_MC_{n}_{K}_{sigma}_{N}.csv")
        

        return control, np.exp(-r * T) * (np.mean(payoff_a))

    option_values = np.exp(-r * T) * np.asarray(payoff)
    #print(option_values)
    data = {"Values": option_values, "Simulation":sim}
    df = pd.DataFrame(data) 
    df.to_csv(f"asian_MC_{n}_{K}_{sigma}_{N}.csv")
    return np.exp(-r * T) * np.mean(payoff), np.std(payoff)/sqrt(n)
>>>>>>> 032ba89957e7c380e10ab087571f80d57358fab3

#%%
#### ASIAN OPTION
K = 99
S = 100
r = 0.06
sigma = 0.2
T = 1
N = 365
M = n = 1000

<<<<<<< HEAD
anal = asian_anal(S,N,T,sigma,r,K)

geom = asian_MC(S,N,T,r,K, n, a = "geometric")
print(anal, geom)
# asian_chris = Asian_call_MC(M=50,S0=100,K=99,T=1,r=0.06,sigma=0.2)
# print(asian_chris)

#%%
nn = [100,500, 1000,5000]
#nn = [100,500, 1000]
=======
asian_analytical = asian_anal(S,N,T,sigma,r,K)
print(asian_analytical)
asian_geom = asian_MC(S,N,T,sigma,r,K, n, type_op = "geometric", param="None")
print(asian_geom)


#%%
nn = [100,500, 1000,5000, 10000, 50000, 100000, 500000]
#nn = [100000]
>>>>>>> 032ba89957e7c380e10ab087571f80d57358fab3
asian_MC_list = []
asian_anal_list = []
standard_error = []
columns = ["Values", "Simulation"]
df_final = pd.DataFrame()
for n in nn:
<<<<<<< HEAD
    
    MC = asian_MC(S,N,T,r,K,n,a = "geometric")
    asian_MC_list.append(MC[0])
    standard_error.append(MC[1])
    asian_anal_list.append(asian_anal(S,N,T,sigma,r,K))

=======
    MC = asian_MC(S,N,T,sigma,r,K,n,type_op = "geometric", param="None")
    #asian_MC_list.append(MC[0])
    #standard_error.append(MC[1])
    asian_anal_list.append(asian_anal(S,N,T,sigma,r,K))

#%%
nn = [100,500, 1000,5000, 10000, 50000, 100000, 500000]
>>>>>>> 032ba89957e7c380e10ab087571f80d57358fab3
frames = [ pd.read_csv(f"asian_MC_{n}.csv") for n in nn ]
result = pd.concat(frames)
print(result)
result.to_csv("asian_MC_final")
#%%
"""
1.1: plot for comparing analytical and MC values
"""
<<<<<<< HEAD
df = pd.read_csv("jToverN/asian_MC_final")
sns.lineplot(data=df, x="Simulation", y="Values", label = "Monte Carlo")
plt.plot(nn, asian_anal_list, label = "Analytical")
plt.show()
=======

#df = pd.read_csv("jToverN/asian_MC_final")
sns.lineplot(data=result, x="Simulation", y="Values", label = "Monte Carlo")
plt.plot(nn, asian_anal_list, label = "Analytical")
plt.xscale("log")
plt.legend()
>>>>>>> 032ba89957e7c380e10ab087571f80d57358fab3
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
the value of the Asian op- tion based on arithmetic averages.
"""
K = 99
S = 100
r = 0.06
sigma = 0.2
T = 1
N = 365
n = 10000

<<<<<<< HEAD
asian_arith_MC = asian_MC(S,N,T,r,K, n, type = "arithmetic")
asian_geom_MC = asian_MC(S,N,T,r,K, n, type = "geometric")
asian_anal = asian_anal(S,N,T,sigma,r,K)
asian_cv = asian_arith_MC + asian_anal + asian_geom_MC
#%%
"""
3.3.b: different parameter settings.
"""
# strike
K = np.linspace(50, 99, 99-50)
# number of paths
N = np.linspace(2, 365*2, 365*2-1)
# number of time points
=======
MC = asian_MC(S,N,T,sigma,r,K, n, type_op="control")
contr = MC[0]
asian_mont = MC[1]

print(contr, asian_mont)
>>>>>>> 032ba89957e7c380e10ab087571f80d57358fab3


#%%
"""
3.3.b: different parameter settings.
"""
K = 99
S = 100
r = 0.06
sigma = 0.2
T = 1
N = 365
n = 100

# strike
control = []
asian_mont = []
k = np.linspace(50, 99, 99-50+1)
for K in k:
    MC = asian_MC(S,N,T,sigma,r,K, n, type_op = "control", param="K")
    control.append(MC[0])
    asian_mont.append(MC[1])

frames = [ pd.read_csv(f"K/asian_MC_{n}_{K}_{sigma}_{N}.csv") for K in k ]
result = pd.concat(frames)
print(result)
result.to_csv("asian_MC_final_K")

sns.lineplot(data=result, x="K", y="Arithmetic", label = "Control variate")
sns.lineplot(data=result, x="K", y="Control", label = "MC only")
plt.legend()
plt.savefig("Asian_3_31.pdf")
plt.show()

#%%
# number of paths
K = 99
S = 100
r = 0.06
sigma = 0.2
T = 1
N = 365
n = 100
NN = np.linspace(2, 365, 365-2+1)
control = []
asian_mont = []

for N in NN:
    MC = asian_MC(S,N,T,sigma,r,K, n, type_op = "control", param="N")
    control.append(MC[0])
    asian_mont.append(MC[1])

frames = [ pd.read_csv(f"N/asian_MC_{n}_{K}_{sigma}_{N}.csv") for N in NN ]
result = pd.concat(frames)
print(result)
result.to_csv("N/asian_MC_final_N")

sns.lineplot(data=result, x="N", y="Arithmetic", label = "Control variate")
sns.lineplot(data=result, x="N", y="Control", label = "MC only")
plt.legend()
plt.savefig("Asian_3_32.pdf")
plt.show()

#%%
# sigma
K = 99
S = 100
r = 0.06
sigma = 0.2
T = 1
N = 365
n = 100
vol = np.linspace(0.0001, 5, 50)
control = []
asian_mont = []

for sigma in vol:
    MC = asian_MC(S,N,T,sigma,r,K, n, type_op = "control", param="sigma")
    control.append(MC[0])
    asian_mont.append(MC[1])

frames = [ pd.read_csv(f"sigma/asian_MC_{n}_{K}_{sigma}_{N}.csv") for sigma in vol ]
result = pd.concat(frames)
print(result)
result.to_csv("sigma/asian_MC_final_sigma")

sns.lineplot(data=result, x="sigma", y="Arithmetic", label = "Control variate")
sns.lineplot(data=result, x="sigma", y="Control", label = "MC only")
plt.legend()
plt.savefig("Asian_3_33.pdf")
plt.show()
#%%