import numpy as np
from array import  *
import sys
import os
from matplotlib import pyplot as plt
import scipy.integrate as integrate
import math
import matplotlib
from scipy.integrate import odeint


def initial(x):
    if(x<-3): return 1+1
    elif(-3<=x<3): return -x/3  +1
    else: return - 1+1
    
C = 1
N = 500
xmin = -4
xmax = 4
dx = ( xmax - xmin ) / (N - 1)
dt = C * dx
a0 = 1

T = 4
NIterations = T/dt

x = np.linspace(xmin, xmax, N)
u = np.empty_like(x)
F = np.empty_like(x)
u_n = np.empty_like(x)

s = 0.1
for i in range(len(x)): u[i] = initial(x[i])

plt.plot(x, u)
plt.show()

i = 0
NIterationsImages = 10
while i < NIterations:
    
    F[:] = 1/2 * u[:] * u[:]
    
    for j in range(1, N-1):
        u_n[j] = 1/2 * ( u[j+1] + u[j-1] ) - dt / ( 2 * dx ) * ( F[j+1] - F[j-1] )
     
    u_n[0] = u_n[1]
    u_n[-1] = u_n[-2]
    
    u[:] = u_n[:]
    
    i = i + 1
    
    if(i%NIterationsImages == 0):
        plt.cla()
        plt.xlim(-4,4)
        plt.ylim(0.,1.05)
        plt.plot(x, u, color='black', linewidth = 1)
        plt.plot(x, np.exp(-(x[:]-i*dt)*(x[:]-i*dt)/s*s), color = "red", linewidth = 1)
        plt.title(str(round(i*dt,3)))
        plt.pause(0.001)
        
