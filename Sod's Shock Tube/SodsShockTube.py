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

def Flux(u):

    P = PolytropicPressureEnergy(u[0], u[1], u[2], gamma)
    F0 = (u[0][:]*u[1][:])
    F1 = (u[0][:]*u[1][:]*u[1][:] + P[:])
    F2 = (u[1][:] * ( u[2][:] + P[:] ))
    
    return F0, F1, F2
  
    
C = 0.1
N = 2000
xmin = 0
xmax = 1
dx = ( xmax - xmin ) / (N - 1)
dt = C * dx
gamma = 1.4

T = 0.2
NIterations = T/dt

x = np.linspace(xmin, xmax, N)

rho   = np.empty_like(x)
p     = np.empty_like(x)
E     = np.empty_like(x)  

rho_n   = np.empty_like(x)
p_n     = np.empty_like(x)
E_n     = np.empty_like(x)

v     = np.empty_like(x)
P     = np.empty_like(x)
F_rho = np.empty_like(x)
F_p   = np.empty_like(x)
F_E   = np.empty_like(x)

rho = InitialDistribution(x , SodTubeDensity)
v   = InitialDistribution(x, zero)
E   = InitialDistribution(x, SodTubeEnergy)

i = 0
while i < NIterations:
    
    p[:] = v[:] * rho[:]
    P = PolytropicPressureEnergy(rho, v, E, gamma)
    F_rho[:] = rho[:]*v[:]
    F_p[:] =  rho[:]*v[:]*v[:] + P[:]
    F_E[:] =  v[:] * ( E[:] + P[:] ) 
    
    
    for j in range(1, N-1):
        rho_n[j] = 1/2 * ( rho[j+1] + rho[j-1] ) - dt / ( 2 * dx ) * ( F_rho[j+1] - F_rho[j-1] )
        p_n[j]   = 1/2 * ( p[j+1]   + p[j-1]   ) - dt / ( 2 * dx ) * ( F_p[j+1]   - F_p[j-1]   )
        E_n[j]   = 1/2 * ( E[j+1]   + E[j-1]   ) - dt / ( 2 * dx ) * ( F_E[j+1]   - F_E[j-1]   )
        
    rho_n[0] = rho_n[1]
    rho_n[-1] = rho_n[-2]
    p_n[0] = p_n[1]
    p_n[-1] = p_n[-2]
    E_n[0] = E_n[1]
    E_n[-1] = E_n[-2]
    
    rho[:] = rho_n[:]
    p[:]   = p_n[:]
    v[:]   = p_n[:] / rho_n[:]
    E[:]   = E_n[:]
    
    i = i + 1
    
    plt.cla()
    plt.xlim(0,1)
    plt.ylim(-1.,1.05)
    plt.scatter(x, rho, color='black', linewidth=0.005)
    plt.scatter(x, PolytropicPressure(rho, gamma), color='red', linewidth=0.005)
    plt.scatter(x, v,   color='black', linewidth=0.005, linestyle = "dashed")
    plt.title(str(round(i*dt,3)))
    plt.pause(0.001)
    
solution = open("solutionSodShockTube.txt", "w")
for j in range(len(x)):
    solution.write("%.8e %.8e %.8e \n" % (x[j], rho[j], v[j]))
solution.close()



