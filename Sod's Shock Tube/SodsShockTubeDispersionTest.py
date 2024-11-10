import numpy as np
from array import  *
import sys
import os
from matplotlib import pyplot as plt
import scipy.integrate as integrate
import math
import matplotlib
from scipy.integrate import odeint

def SodTubeDensity(x):
    rho = np.empty_like(x)
    for j in range(len(x)):
        if(x[j] < 0.5 * (x[-1]+x[0])): rho[j] = 1
        else: rho[j] = 0.125
    return rho

def SodTubeEnergy(x):
    E = np.empty_like(x)
    for j in range(len(x)):
        if(x[j] < 0.5 * (x[-1]+x[0])): E[j] = 2.5
        else: E[j] = 0.25
    return E

def zero(x):
    return np.zeros(len(x))

def InitialDistribution(x, function):
    return function(x)

def PolytropicPressure(rho, gamma):
    return rho**gamma

def PolytropicPressureEnergy(rho, v, E, gamma):
    return (gamma-1) * (E - 1/2 * rho * v * v) 
    
f = 2
n = 3
C0 = 0.35
C = np.zeros(n)
for j in range(n):
    C[j] = C0 / f**j

N = 1000
xmin = 0
xmax = 1
T = 1
dx = ( xmax - xmin ) / (N - 1)
dt = np.zeros(n)
NIterations = np.zeros(n)
NIterations0 = 10000
for j in range(n):
    dt[j] = C[j] * dx
    NIterations[j] = int(NIterations0*f**j)
gamma = 1.4

print(NIterations, dt)


x = np.linspace(xmin, xmax, N)

rho   = []
p     = []
E     = []

rho_n   = []
p_n     = []
E_n     = []

v     = []
P     = []
F_rho = []
F_p   = []
F_E   = []

T     = np.zeros(n)

for j in range(n):
    rho.append(np.empty_like(x))
    p.append(np.empty_like(x))
    E.append(np.empty_like(x))
    
    rho_n   .append(np.empty_like(x))
    p_n     .append(np.empty_like(x))
    E_n     .append(np.empty_like(x))

    v     .append(np.empty_like(x))
    P     .append(np.empty_like(x))
    F_rho .append(np.empty_like(x))
    F_p   .append(np.empty_like(x))
    F_E   .append(np.empty_like(x))

    rho[j] = InitialDistribution(x , SodTubeDensity)
    v[j]   = InitialDistribution(x, zero)
    E[j]   = InitialDistribution(x, SodTubeEnergy)

i = 0
styles = ["solid", "dashed", "dotted"]
NIterationsImages = 20

while i < NIterations[-1]:
    for k in range(n): 
        if(i%f**(n-k) ==0):
            p[k][:] = v[k][:] * rho[k][:]
            P[k] = PolytropicPressureEnergy(rho[k], v[k], E[k], gamma)
            F_rho[k][:] = rho[k][:]*v[k][:]
            F_p[k][:] =  rho[k][:]*v[k][:]*v[k][:] + P[k][:]
            F_E[k][:] =  v[k][:] * ( E[k][:] + P[k][:] ) 
        
            for j in range(1, N-1):
                rho_n[k][j] = 1/2 * ( rho[k][j+1] + rho[k][j-1] ) - dt[k] / ( 2 * dx ) * ( F_rho[k][j+1] - F_rho[k][j-1] )
                p_n[k][j]   = 1/2 * ( p[k][j+1]   + p[k][j-1]   ) - dt[k] / ( 2 * dx ) * ( F_p[k][j+1]   - F_p[k][j-1]   )
                E_n[k][j]   = 1/2 * ( E[k][j+1]   + E[k][j-1]   ) - dt[k] / ( 2 * dx ) * ( F_E[k][j+1]   - F_E[k][j-1]   )
            
            rho_n[k][0] = rho_n[k][1]
            rho_n[k][-1] = rho_n[k][-2]
            p_n[k][0] = p_n[k][1]
            p_n[k][-1] = p_n[k][-2]
            E_n[k][0] = E_n[k][1]
            E_n[k][-1] = E_n[k][-2]
            
            rho[k][:] = rho_n[k][:]
            p[k][:]   = p_n[k][:]
            v[k][:]   = p_n[k][:] / rho_n[k][:]
            E[k][:]   = E_n[k][:]
            
            T[k] = T[k] + dt[k]
    
    if( i%f**(n-1) ==0 and i%NIterationsImages == 0 ):
        print(T)
        plt.cla()
        plt.xlim(0,1)
        plt.ylim(0,1.05)
        for k in range(n):
            plt.plot(x, rho[k], color='black', linewidth=1.0, linestyle = styles[k], label = str(1/f**k))
        plt.legend()
        plt.title(str(round(T[0],2)))
        plt.pause(0.001)
        
    i = i + 1
    


