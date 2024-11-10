import numpy as np
from array import  *
import sys
import os
from matplotlib import pyplot as plt
import scipy.integrate as integrate
import math
import matplotlib
from scipy.integrate import odeint

C = 1
N = 500
xmin = -4
xmax = 4
dx = ( xmax - xmin ) / (N - 1)
dt = C * dx
a0 = 1

T = 2
NIterations = T/dt

x = np.linspace(xmin, xmax, N)
f = np.empty_like(x)
F = np.empty_like(x)
f_n = np.empty_like(x)

s = 0.1
f[:] = np.exp(-x[:]*x[:]/s*s)

i = 0
NIterationsImages = 5
while i < NIterations:
    
    F[:] = a0 * f[:]
    
    for j in range(1, N-1):
        f_n[j] = 1/2 * ( f[j+1] + f[j-1] ) - dt / ( 2 * dx ) * ( F[j+1] - F[j-1] )
     
    f_n[0] = f_n[1]
    f_n[-1] = f_n[-2]
    
    f[:] = f_n[:]
    
    i = i + 1
    
    if(i%NIterationsImages == 0):
        plt.cla()
        plt.xlim(-4,4)
        plt.ylim(0.,1.05)
        plt.plot(x, f, color='black', linewidth = 0.9)
        plt.plot(x, np.exp(-(x[:]-i*dt)*(x[:]-i*dt)/s*s), color = "red", linewidth = 0.9)
        plt.title(str(round(i*dt,3)))
        plt.pause(0.001)
        
