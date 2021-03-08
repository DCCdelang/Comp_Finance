#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import *
from scipy.stats import norm, gmean
import seaborn as sns

K = 99
S = 100
r = 0.06
sigma = 0.2
T = 1
N = 50
#%%

def option_price(K, S, r, sigma, T, pay_offs):
    return np.exp(-r * T)*(np.mean(pay_offs))

def ST(K, S, r, sigma, T):
   
    Z = np.random.normal()
    ST = S* (np.exp( (r-0.5*(sigma**2))*T + sigma*np.sqrt(T)*Z))
    
    return max(K - ST, 0)

def digital(K, S, r, sigma, T):
   
    Z = np.random.normal()
    ST = S* (np.exp( (r-0.5*(sigma**2))*T + sigma*np.sqrt(T)*Z))
    
    if max(K - ST, 0) > 0 :
        return 1, ST
    else:
        return 0, ST

def N_(x):
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

def asian_MC(S,N,T,r,K, n, a = "geometric"):
    payoff = []
    sim = []
    for _ in range(n):
        S_ti_ar = 0

        ST=S

        if a == "arithmetic":
            Z = np.random.normal()
            for j in range(N):
                T_i = T/N
                ST = ST * (np.exp( (r-0.5*sigma**2)*T_i + sigma*np.sqrt(T_i)*Z))
                S_ti_ar.append(ST)
        
            payoff.append(max(np.mean(S_ti_ar)-K, 0))
            sim.append(n)

        elif a == "geometric":
            S_ti_geo = [] 
            for j in range(N):
                Z = np.random.normal(0,1)
                T_i = T/N
                ST = ST * np.exp( (r-0.5*sigma**2)*T_i + sigma*np.sqrt(T_i)*Z)
                S_ti_geo.append(ST)

            payoff.append(max(gmean(S_ti_geo)-K, 0))
            sim.append(n)
    
    data = {"Values":payoff, "Simulation":sim}
    df = pd.DataFrame(data) 
    df.to_csv(f"asian_MC_{n}.csv")
    return np.exp(-r * T) * np.mean(payoff), np.std(payoff)/sqrt(N)

"""
1.1 Carry out convergence studies by increasingthe number of trials.  How do your results compare with the results obtained in assignment 1?
"""
print(black_scholes_p(S,N,T,sigma,r,K))


n = [100,500, 1000,5000, 10000, 50000, 100000, 500000, 1000000]
a = [(black_scholes_p(S,N,T,sigma,r,K))]*len(n)
data_bs = {"Values":a, "Price":n}
values = []
time = []
for i in range(10):
    for path in n:
        value = []
        for i in range(path):
            approxList = ST(K, S, r, sigma, T)
            # print(approxList)
            value.append(approxList)
        time.append(path)
        values.append(option_price(K, S, r, sigma, T, value))

data = {"Values":values, "Price":time}
df = pd.DataFrame(data) 
df.to_csv(f"Monte_carlo.csv")
# value = []
# value2 = []
# time = []

data = pd.read_csv(f'Monte_carlo.csv')
# sns.lineplot(data=data, x="Price", y="Values")
# plt.xscale("log") 
# # plt.show()
# data = pd.read_csv(f'Monte_carlo.csv')


sns.lineplot(data=data, x="Price", y="Values", label ="Monte Carlo")
sns.lineplot(data=data_bs, x="Price", y="Values", label ="Analytical")
plt.xlabel("Paths", fontsize=14)
plt.ylabel("Option price", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xscale("log")
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig("Convergence_option_price.pdf")
plt.show()

data = pd.read_csv(f'Monte_carlo.csv')
values = data.loc[data['Price'] == 1000000]["Values"]
m = np.mean(values)
s = np.std(values)
se = s/sqrt(100)

ci = [m+(se*1.96), m-(se*1.96)]

ci2 = [m+(s*1.96), m-(s*1.96)]
print(f"SE = {se} sd = {s} ci1 = {ci} ci2 = {ci2}")
value = []
for i in range(1000000):
    approxList = ST(K, S, r, sigma, T)
    # print(approxList)
    value.append(np.exp(-r * T)* approxList)
# time.append(N)
m = np.mean(value)
sd = np.std(value)

se = sd/np.sqrt(1000000)

ci = [m+(se*1.96), m-(se*1.96)]
print(f"CI3 = {ci}, sd = {sd}, se = {se} mean = {m}")
option_prices.append(option_price(K, S, r, sigma, T, value))


plt.plot(n, option_prices)
data = {"Values":option_prices, "Price":time}
df = pd.DataFrame(data) 
df.to_csv(f"Monte_carlo.csv")
sns.lineplot(data=df, x="Time", y="Values")
plt.xscale("log") 
plt.savefig("Convergence_option_price.pdf")
plt.show()



"""
1.1 Perform numerical tests for varying values for the strike and the volatility parameter.
"""
K = 99
S = 100
r = 0.06
sigma = 0.2
T = 1
N = 50


option_prices = []
time = []
std_error = []
mean = []
confidence_interval = []

sigmas = np.linspace(0.1, 1, 10)
n = 1000000


for sigma in list(sigmas):
    print(sigma)
    value = []
    value2 = []
    for i in range(n):
        approxList = ST(K, S, r, sigma, T)
        value2.append(np.exp(-r * T)* approxList)
        value.append(approxList)
    time.append(sigma)
    option_prices.append(option_price(K, S, r, sigma, T, value))   
    m = np.mean(value2)  
    
    se = np.std(value2)/np.sqrt(n)
  
    confidence_interval.append([round(m-(se*1.96),3), round(m+(se*1.96),3)])
    std_error.append(round(se,3))
    mean.append(round(m,3))
data = {"Values":option_prices, "Sigma":time, "Mean":mean, "Confidence interval":confidence_interval, "Standard error": std_error}

df.to_csv(f"Monte_carlo_volatility.csv")

print(option_prices)

plt.plot(sigmas, option_prices)
plt.show()
#%%
"""
1.1 Perform numerical tests for varying values for the strike and the strike parameter.
"""
K = 99
S = 100
r = 0.06
sigma = 0.2
T = 1
N = 50


Ks = np.linspace(95, 105, 11)
option_prices = []
time = []

std_error = []
mean = []
ste = []
confidence_interval = []
n = 1000000


for K in list(Ks):
    print(K)
    value = []
    value2 = []
    for i in range(n):
        approxList = ST(K, S, r, sigma, T)
        value2.append(np.exp(-r * T)* approxList)
        value.append(approxList)
    time.append(K)
    option_prices.append(option_price(K, S, r, sigma, T, value))     
    m = np.mean(value2)   
    se = np.std(value2)/np.sqrt(n)
    confidence_interval.append([round(m-(se*1.96),3), round(m+(se*1.96),3)])
    std_error.append(round(se,3))
    mean.append(round(m,3))
print(se)
data = {"Values":option_prices, "Sigma":time, "Mean":mean, "Confidence interval":confidence_interval, "Standard error":std_error}
df = pd.DataFrame(data) 
df.to_csv(f"Monte_carlo_strike.csv")


"""
2.1 
# """
# # Initial values
K = 99
S = 100
r = 0.06
sigma = 0.2
T = 1
N=50

# Initial lists
option_prices = []
option_prices2 = []
value = []
value2 = []
w = []
# Paths and epsilons
paths = [100000]
epsilons = [0.1]

# Black-Scholes model
d = ((np.log(S/K) + ((r+(sigma**2)/2)) * T )/(sigma*np.sqrt(T)))

d_1 = d - sigma* np.sqrt(T)

d_s = -(N_(-d))

print("\n")
print("Black-Scholes = ", d_s)
error1 = []
error2 = []
# Unbumped
# np.random.seed(42)
# for epsilon in epsilons:
#     for path in paths:
        for _ in range(path):
            approxList = ST(K, S, r, sigma, T)
            value.append(approxList)
        w.append((epsilon, path))    
        option_prices.append(option_price(K, S, r, sigma, T, value))

# Bumped
# np.random.seed(42)
for epsilon in epsilons:
    for path in paths:
        for _ in range(path):
            approx_list2 = ST(K, S + epsilon, r, sigma, T)
            value2.append(approx_list2)
        option_prices2.append(option_price(K, S + epsilon, r, sigma, T, value2))

for i in range(len(value)):
    error = value2[i] - value[i]
    error1.append(error)
se = (np.std(error1)/(np.sqrt(paths[0]))/epsilons[0]
print("Standard error = ", abs(se))
# print(option_prices, option_prices2)
# delta = (option_prices2[0] - option_prices[0])/epsilon
# print("Bumped and revalue = ",delta)
for i in range(len(option_prices)):
    print(f"paths = {w[i][1]} epsilone = {w[i][0]}")
    delta = (option_prices2[i] - option_prices[i])/epsilons[0]
    print(round(delta,4))


digital_analytical = - (norm.pdf(d_1)/(sigma*S*np.sqrt(T))) *np.exp(-r*T)
print(digital_analytical)

# Unbumped
np.random.seed(42)
for _ in range(paths):
    approxList = digital(K, S, r, sigma, T)[0]
    value.append(approxList)
option_prices.append(option_price(K, S, r, sigma, T, value))

# Bumped
np.random.seed(42)
for _ in range(paths):
    approx_list2 = digital(K, S + epsilon, r, sigma, T)[0]
    value2.append(approx_list2)
option_prices2.append(option_price(K, S + epsilon, r, sigma, T, value2))


print(option_prices, option_prices2)
delta = (option_prices2[0] - option_prices[0])/epsilon
print("Bumped and revalue = ",delta)


# # # Pathwise
p = 1000000
tot_delta = 0
paths1 = []
for _ in range (p):
    # np.random.seed(42)
    digital_ = digital(K, S, r, sigma, T)
    # print(digital_)
    I = digital_[0]
    ST = digital_[1]
    deri = norm.pdf(ST, loc=99, scale=1)
    deri_s = ST/S
    delta = deri_s*deri
    paths1.append(delta)
end_delta = np.exp(-r*T)*np.sum(paths1)/p
se = np.std(paths1)/np.sqrt(p)
print("SE pathwise = ", se)
print("Delta pathwise = ", end_delta)


# Likelihood ratio method
def eulerMethod_Z(S,T,K,r,vol):
    Z = np.random.normal()
    S_T = S * np.exp((r-0.5*(vol**2))*T + vol*(T**0.5)*Z)
    return S_T, Z
 
def binary_put_payoff(S_T, K):
    if S_T < K:
        return 1.0
    else:
        return 0.0
 
N_samples = 1000000
 
# Likelihood ratio method for binary digital delta approximation for european put
delta_list = []
# np.random.seed(42)
for i in range(N_samples):
    S_T, Z = eulerMethod_Z(S,T,K,r,sigma)
    binary_payoff = binary_put_payoff(S_T, K)
    delta_list.append(np.exp(-r*T)*binary_payoff*(Z/(S*sigma*T**0.5)))
delta = np.mean(delta_list)
se = np.std(delta_list)/np.sqrt(N_samples)
print("SE pathwise = ", se)
print("Delta:", delta)
 
# print((delta-binary_BS_Put)/binary_BS_Put)