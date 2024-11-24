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

def Flux_Euler(u):
    P = PolytropicPressureEnergy(u[0], u[1]/u[0], u[2], gamma)
    F0 = (u[1][:])
    F1 = (u[1][:]*u[1][:]/u[0][:] + P[:])
    F2 = (u[1][:]/u[0][:] * ( u[2][:] + P[:] ))
    return [F0, F1, F2]

def CalculateNextIteration(dx, dt, U, Flux):
    Fluxes = Flux(U) 
    next = []
    for i in range(len(U)):
        next.append(np.zeros_like(U[i]))
        for j in range(1, len(U[0])-1):
            next[i][j] = 1/2 * ( U[i][j+1] + U[i][j-1] ) - dt / ( 2 * dx ) * ( Fluxes[i][j+1]   - Fluxes[i][j-1] )
        next[i][0] = next[i][1]
        next[i][-1] = next[i][-2]
    for i in range(len(U)): U[i][:] = next[i][:]
    return U
    
C = 0.3
N = 500
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
v     = np.empty_like(x)

rho = InitialDistribution(x , SodTubeDensity)
v   = InitialDistribution(x, zero)
E   = InitialDistribution(x, SodTubeEnergy)

p[:] = v[:] * rho[:]
i = 0

while i < NIterations:

    rho, p, E = CalculateNextIteration(dx, dt, [rho, p, E], Flux_Euler)
    
    i = i + 1
    
    plt.cla()
    plt.xlim(0,1)
    plt.ylim(-1.,1.05)
    plt.scatter(x, rho, color='black', linewidth=0.005)
    plt.title(str(round(i*dt,3)))
    plt.pause(0.001)
    
solution = open("solutionSodShockTube.txt", "w")
for j in range(len(x)):
    solution.write("%.8e %.8e %.8e \n" % (x[j], rho[j], v[j]))
solution.close()



