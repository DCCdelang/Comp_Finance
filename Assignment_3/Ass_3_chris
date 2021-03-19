import numpy as np
import matplotlib.pyplot as plt

def OptioncallI(Smax,M,T,N,K,r,sigma):
    
    #discretization
    ds=Smax/float(M)   #step size for the stock
    dt=T/float(N)      #step size for time
    
    i=np.arange(1,M,dtype=np.float)
    
    #initializing the risk neutral probabilities
    
    P=0.5*dt*(r*i-sigma**2*i**2)
    Q=1+ dt*(sigma**2*i**2+r)
    R=-0.5*dt*(sigma**2*i**2+r*i)
    
    A=np.diag(Q)+np.diag(P[1:],k=-1)+np.diag(R[0:M-2],k=1) #the tri-diagonal matrix
    
    F=np.zeros((N+1,M+1))  #The matrix for the option value
    
    #Boundary conditions
    
    F[N,:]=np.maximum(np.arange(0,Smax+ds/2.0,ds,dtype=np.float)-K,0)
    F[:,0]=0
    F[:,M]=[Smax * np.exp(-r*( N - j)*dt) for j in range(N+1)]
    
    for j in range(N-1,-1,-1):
        d=F[j+1,1:M] 
        d[0]=d[0]-P[0]*F[j,0]   #inserts the first value
        d[M-2]=d[M-2]-R[M-2]*F[j,M]   #inserts the last value
        #print("d is: ",d.shape)

        F[j,1:M]=np.linalg.solve(A,d) #solves the simultaneous equations
        F[j,:]=np.maximum(np.arange(0,Smax+ds/2.0,ds,dtype=np.float)-K,F[j,:]) #comparison
    print("F is: ",F.shape)
    return F, F[0,(M+1)//2] #returns the option value
option = OptioncallI(120,50,5/12.0,100,99,0.06,0.2)
print ("The value of an American call option is" , option[1])
#print("The matrix is: ",option[0])

from matplotlib import cm 


y = np.linspace(0,2/12.0,101)
x= np.linspace(0,150,51)
F= option[0]

X, Y = np.meshgrid(x, y)
print("X.shape is: ", X.shape)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, F, 50, cmap=cm.cool)
ax.set_xlabel('S')
ax.set_ylabel('t')
ax.set_zlabel('option price')
plt.show()
