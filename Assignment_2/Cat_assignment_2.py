#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import *
from scipy.stats import norm, gmean, variance
import seaborn as sns

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

def asian_MC(S,N,T,r,K, n, type_op = "geometric"):
    payoff = []
    payoff_a = []
    payoff_g = []
    sim = []
    #data = {"Values":payoff, "Simulation":sim}
    #df = pd.DataFrame() 
    for i in range(n):
        S_ti_ar = []
        S_ti_geo = []
        ST = ST_g = ST_a = S
        T_i = T/N

        if type_op == "arithmetic":
            for j in range(N):
                Z = np.random.normal()
                ST = ST + (np.exp( (r-0.5*sigma**2)*T_i + sigma*np.sqrt(T_i)*Z))
                S_ti_ar.append(ST)
        
            payoff.append(max(np.mean(S_ti_ar)-K, 0))
            sim.append(n)

        elif type_op == "geometric":
            for j in range(N):
                Z = np.random.normal(0,1)
                ST = ST * np.exp( (r-0.5*sigma**2)*T_i + sigma*np.sqrt(T_i)*Z)
                S_ti_geo.append(ST)


            payoff.append(max(gmean(S_ti_geo)-K, 0))
            sim.append(n)

        elif type_op == "control":
            for j in range(N):
                Z = np.random.normal(0,1)
                
                ST_g = ST_g * (np.exp( (r-0.5*sigma**2)*T_i + sigma*np.sqrt(T_i)*Z))
                S_ti_geo.append(ST_g)
                
                ST_a = ST_a + (np.exp( (r-0.5*sigma**2)*T_i + sigma*np.sqrt(T_i)*Z))
                S_ti_geo.append(ST_a)

            payoff_g.append(max(gmean(S_ti_geo)-K, 0))            
            payoff_a.append(max(np.mean(S_ti_geo)-K, 0))            
            sim.append(n)


    if type_op == "control":

        control = np.exp(-r * T) * (np.mean(payoff_a)+asian_anal(S,N,T,sigma,r,K)-np.mean(payoff_g))
        var_contr = np.exp(-r * T)**2 * ( variance(payoff_a) + variance(payoff_g) - 2*np.cov(payoff_a, payoff_g))
        var_MC = variance(payoff_a)
        data = {"Values":payoff, "Simulation":sim}
        df = pd.DataFrame(data) 
        df.to_csv(f"asian_MC_{n}.csv")

        return control, var_contr, np.exp(-r * T) * (np.mean(payoff_a)), var_MC


    data = {"Values":payoff, "Simulation":sim}
    df = pd.DataFrame(data) 
    df.to_csv(f"asian_MC_{n}.csv")
    return np.exp(-r * T) * np.mean(payoff), np.std(payoff)/sqrt(n)

def exact_GBM(S0=100,K=99,T=1,r=0.06,sigma=0.2,N=365):
    dt= T/N
    S=[] # history stock price
    for n in range(N):
        Zm= np.random.normal(0,1)
        S0= S0 * np.exp((r-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*Zm)
        S.append(S0)
    return S
    
def Asian_call_MC(M:int,S0=100,K=99,T=1,r=0.06,sigma=0.2):
    payoff=[]
    for i in range(0,M):
        S=exact_GBM(S0,K,T,r,sigma) # all the history price
        payoff.append(max(gmean(S)-K, 0))
    return np.exp(-r*T)*np.mean(payoff), np.std(payoff)/np.sqrt(M)

#%%
#### ASIAN OPTION
K = 99
S = 100
r = 0.06
sigma = 0.2
T = 1
N = 365
M = n = 1000

anal = asian_anal(S,N,T,sigma,r,K)
geom = asian_MC(S,N,T,r,K, n, type_op = "geometric")
print(anal, geom)
# asian_chris = Asian_call_MC(M=50,S0=100,K=99,T=1,r=0.06,sigma=0.2)
# print(asian_chris)

#%%
nn = [100,500, 1000,5000]
#nn = [100,500, 1000]

asian_analytical = asian_anal(S,N,T,sigma,r,K)
print(asian_analytical)
asian_geom = asian_MC(S,N,T,r,K, n, type_op = "geometric")
print(asian_geom)
#asian_chris = Asian_call_MC(M=50,S0=100,K=99,T=1,r=0.06,sigma=0.2)
#print(asian_chris)

#%%
nn = [100,500, 1000,5000, 10000, 50000, 100000, 500000, 1000000]
#nn = [100,500, 1000,5000, 10000, 50000]
asian_MC_list = []
asian_anal_list = []
standard_error = []
asian_chris = []
columns = ["Values", "Simulation"]
df_final = pd.DataFrame()
for n in nn:
    MC = asian_MC(S,N,T,r,K,n,type_op = "geometric")
    asian_MC_list.append(MC[0])
    standard_error.append(MC[1])
    asian_chris.append(Asian_call_MC(M=n,S0=100,K=99,T=1,r=0.06,sigma=0.2))
    asian_anal_list.append(asian_anal(S,N,T,sigma,r,K))

#%%
frames = [ pd.read_csv(f"correct_asian_MC/asian_MC_{n}.csv") for n in nn ]
result = pd.concat(frames)
print(result)
result.to_csv("asian_MC_final")
#%%
"""
1.1: plot for comparing analytical and MC values
"""
#df = pd.read_csv("jToverN/asian_MC_final")
sns.lineplot(data=result, x="Simulation", y="Values", label = "Monte Carlo")
plt.plot(nn, asian_anal_list, label = "Analytical")
#plt.plot(nn, asian_chris, label = "Chris")
plt.xscale("log") 
plt.legend()
plt.show()
plt.savefig("Asian_1_2.pdf")
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
M = n = 1000

control = asian_MC(S,N,T,r,K, n, type_op = "control")[0]
var_contr = asian_MC(S,N,T,r,K, n, type_op = "control")[1]
asian_MC = sian_MC(S,N,T,r,K, n, type_op = "control")[2]
var_MC = sian_MC(S,N,T,r,K, n, type_op = "control")[3]
print(control, var_contr, asian_MC, var_MC)


#%%
"""
3.3.b: different parameter settings.
"""
asian_geom_MC = []
asian_ana
# strike
K = np.linspace(50, 99, 99-50)
for k in K:
    asian_arith_MC = asian_MC(S,N,T,r,K, n, type_op = "arithmetic")[0]
    asian_geom_MC = asian_MC(S,N,T,r,K, n, type_op = "geometric")[0]
    asian_anal = asian_anal(S,N,T,sigma,r,K)
    asian_cv = asian_arith_MC + asian_anal + asian_geom_MC
# number of paths
N = np.linspace(2, 365*2, 365*2-1)
# number of time points


#%%
"""
1.1 Carry out convergence studies by increasingthe number of trials.  
How do your results compare with the results obtained in assignment 1?
"""
print(black_scholes_p(S,N,T,sigma,r,K))

K = 99
S = 100
r = 0.06
sigma = 0.2
T = 1

option_prices = []
value = []
time = []
n = [100,500, 1000,5000, 10000, 50000, 100000, 500000, 1000000]

for i in range(5):
    for N in list(n):
        for i in range(int(N)):
            approxList = ST(K, S, r, sigma, T)
            # print(approxList)
            value.append(approxList)
        time.append(N)
        option_prices.append(option_price(K, S, r, sigma, T, value))


# plt.plot(n, option_prices)
data = {"Values":option_prices, "Time":time}
df = pd.DataFrame(data) 
df.to_csv(f"Monte_carlo.csv")
sns.lineplot(data=df, x="Time", y="Values")
plt.xscale("log") 
plt.savefig("Convergence_option_price.pdf")
plt.show()

"""
1.1 Perform numerical tests for varying values for the strike and the volatility parameter.
"""


option_prices = []
value = []
time = []
sigmas = np.linspace(0.01, 10, 10)

for sigma in list(sigmas):
    value = []
    for i in range(1000000):
        approxList = ST(K, S, r, sigma, T)
        # print(approxList)
        value.append(approxList)
    time.append(sigma)
    option_prices.append(option_price(K, S, r, sigma, T, value))
    print(option_prices)


print(option_prices)

plt.plot(sigmas, option_prices)
plt.show()
#%%
option_prices = []

time = []
Ks = np.linspace(80, 120, 10)

for K in list(Ks):
    value = []
    for i in range(1000000):
        approxList = ST(K, S, r, sigma, T)
        # print(approxList)
        value.append(approxList)
    time.append(sigma)
    option_prices.append(option_price(K, S, r, sigma, T, value))

plt.plot(Ks, option_prices)
plt.show()

#%%
K = 99
S = 100
r = 0.06
sigma = 0.2
T = 1
N=50
epsilons = [0.01,0.02,0.5]

option_prices = []

option_prices2 = []
value = []
value2 = []
time = []

for epsilon in epsilons:
    for i in range(1000000):
        approxList = ST(K, S, r, sigma, T)
        approx_list2 = ST(K, S+epsilon, r, sigma, T)
        value.append(approxList)
        value2.append(approx_list2)
    option_prices.append(option_price(K, S, r, sigma, T, value))
    option_prices2.append(option_price(K, S, r, sigma, T, value2))

print(option_prices)

for i in range(3):
    delta = (option_prices2[i] - option_prices[i])/epsilon
    print(delta)
# %%
