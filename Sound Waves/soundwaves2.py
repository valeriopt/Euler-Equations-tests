import numpy as np
from array import  *
import sys
import os
from matplotlib import pyplot as plt
import scipy.integrate as integrate
import math
import matplotlib
from scipy.integrate import odeint

def zero(x):
    one = np.empty_like(x)
    for i in range(len(one)): one[i] = 0.05
    return one

def SineWave(x):
    result = np.empty_like(x)
    for i in range(len(x)): result[i] = 1 + 0.05 * np.sin(2*np.pi*x[i]) 
    return result

def Wave(t,x):
    return 1 + 0. * np.sin(2*np.pi*(x-t))

def InitialDistribution(x, function):
    return function(x)

def PolytropicPressure(rho, gamma):
    return rho**gamma

C = 0.1
N = 2000
xmin = 0
xmax = 1
dx = ( xmax - xmin ) / (N-1)
dt = C * dx
gamma = 1.
NIterationImage = 100

T = 1
NIterations = T/dt

x = np.linspace(xmin, xmax, N)

rho   = np.empty_like(x)
p     = np.empty_like(x)  

rho_n   = np.empty_like(x)
p_n     = np.empty_like(x)

v     = np.empty_like(x)
P     = np.empty_like(x)
F_rho = np.empty_like(x)
F_p   = np.empty_like(x)

rho = InitialDistribution(x , SineWave)
v   = InitialDistribution(x, zero)

i = 0
while i < NIterations:
    
    p[:] = v[:] * rho[:]

    P = PolytropicPressure(rho, gamma)
    F_rho[:] = rho[:]*v[:]
    F_p[:] =  rho[:]*v[:]*v[:] + P[:]
    F_rho_gR = F_rho[0]; F_rho_gL = F_rho[-1]; F_p_gR = F_p[0]; F_p_gL = F_p[-1]  
    
    for j in range(1, N-1):
        rho_n[j] = 1/2 * ( rho[j+1] + rho[j-1] ) - dt / ( 2 * dx ) * ( F_rho[j+1] - F_rho[j-1] )
        p_n[j]   = 1/2 * ( p[j+1]   + p[j-1]   ) - dt / ( 2 * dx ) * ( F_p[j+1]   - F_p[j-1]   )
        
    rho_n[0]  = 1
    rho_n[-1] = 1
    p_n[0] = 0
    p_n[-1] = 0
    
    rho[:] = rho_n[:]
    p[:]   = p_n[:]
    v[:]   = p_n[:] / rho_n[:]
    
    i = i + 1
    
    if(i%NIterationImage==0):
        plt.cla()
        plt.xlim(0,1)
        plt.ylim(-1,1.15)
        plt.plot(x, rho, color='black', linewidth=0.005)
        #plt.scatter(x, PolytropicPressure(rho, gamma), color='red', linewidth=0.005)
        #plt.scatter(x, v,   color='red', linewidth=0.005)
        #plt.plot(x, Wave(i*dt,x)-rho, color='black')
        plt.ylim(-0.1,0.1)
        plt.title(str(round(i*dt,3)))
        plt.pause(0.001)