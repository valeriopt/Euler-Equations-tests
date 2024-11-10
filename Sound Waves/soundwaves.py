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

def Wave(x):
    result = np.empty_like(x)
    for i in range(len(x)): result[i] = 1 + 0.1 * np.sin(2*np.pi*x[i]) 
    return result

def InitialDistribution(x, function):
    return function(x)

def PolytropicPressure(rho, gamma):
    return rho**gamma

C = 0.1
N = 2000
xmin = 0
xmax = 1
dx = ( xmax - xmin ) / (N)
dt = C * dx
gamma = 1.
NIterationImage = 100

T = 1
NIterations = T/dt

x = [i*dx for i in range(N)]
x_gR = x[0]
x_gL = x[-1]

rho   = np.empty_like(x)
p     = np.empty_like(x)  

rho_n   = np.empty_like(x)
p_n     = np.empty_like(x)

v     = np.empty_like(x)
P     = np.empty_like(x)
F_rho = np.empty_like(x)
F_p   = np.empty_like(x)

rho = InitialDistribution(x , Wave)
v   = InitialDistribution(x, zero)


i = 0
while i < NIterations:
    
    p[:] = v[:] * rho[:]
    rho_gR = rho[0]; rho_gL = rho[-1]; p_gR = p[0]; p_gL = p[-1]

    P = PolytropicPressure(rho, gamma)
    F_rho[:] = rho[:]*v[:]
    F_p[:] =  rho[:]*v[:]*v[:] + P[:]
    F_rho_gR = F_rho[0]; F_rho_gL = F_rho[-1]; F_p_gR = F_p[0]; F_p_gL = F_p[-1]  
    
    for j in range(1, N-1):
        rho_n[j] = 1/2 * ( rho[j+1] + rho[j-1] ) - dt / ( 2 * dx ) * ( F_rho[j+1] - F_rho[j-1] )
        p_n[j]   = 1/2 * ( p[j+1]   + p[j-1]   ) - dt / ( 2 * dx ) * ( F_p[j+1]   - F_p[j-1]   )
        
    rho_n[0]  = 1/2 * ( rho[1] + rho_gL )  - dt / ( 2 * dx ) * ( F_rho[1] - F_rho_gL )
    rho_n[-1] = 1/2 * ( rho_gR + rho[-2] ) - dt / ( 2 * dx ) * ( F_rho_gR - F_rho[-2] )
    p_n[0] = 1/2 * ( p[1] + p_gL )  - dt / ( 2 * dx ) * ( F_p[1] - F_p_gL )
    p_n[-1] = 1/2 * ( p_gR + p[-2] ) - dt / ( 2 * dx ) * ( F_p_gR - F_p[-2] )
    
    rho[:] = rho_n[:]
    p[:]   = p_n[:]
    v[:]   = p_n[:] / rho_n[:]
    
    i = i + 1
    
    if(i%NIterationImage==0):
        plt.cla()
        plt.xlim(0,1)
        plt.ylim(-1,1.15)
        plt.scatter(x, rho, color='black', linewidth=0.005)
        #plt.scatter(x, PolytropicPressure(rho, gamma), color='red', linewidth=0.005)
        plt.scatter(x, v,   color='red', linewidth=0.005)
        plt.title(str(round(i*dt,3)))
        plt.pause(0.001)