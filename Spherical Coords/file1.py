import numpy as np
from array import  *
import sys
import os
from matplotlib import pyplot as plt
import scipy.integrate as integrate
import math
import matplotlib
from scipy.integrate import odeint

def ShockDensity(x):
    rho = np.zeros_like(x)
    for j in range(len(x)):
        if(-0.5 <x[j] < 0.5): rho[j] = 1
        else: rho[j] = 0.125
    return rho

def zero(x):
    return np.zeros(len(x))

def InitialDistribution(x, function):
    return function(x)

def PolytropicPressure(rho, gamma):
    return rho**gamma

def PolytropicPressureEnergy(rho, v, E, gamma):
    return (gamma-1) * (E - 1/2 * rho * v * v) 
    
C = 0.1
N = 200
xmin = 0
xmax = 1
dx = ( xmax - xmin ) / (N - 1)
dt = C * dx
gamma = 1.
alpha = 2
NIterationImage = 100

T = 1
NIterations = T/dt

x = np.linspace(xmin, xmax, N) + dx/2
x_gL = x[0] - dx
x_gR = x[0] + dx

print(x[0:5], x_gL)

rho   = np.empty_like(x)
p     = np.empty_like(x)

rho_n   = np.empty_like(x)
p_n     = np.empty_like(x)

v     = np.empty_like(x)
P     = np.empty_like(x)
F_rho = np.empty_like(x)
F_p   = np.empty_like(x)

rho = InitialDistribution(x , ShockDensity)
v   = InitialDistribution(x, zero)

i = 0
while i < NIterations:
    
    p[:] = v[:] * rho[:]
    rho_gL = rho[0]; rho_gR = rho[-1]; p_gL = -p[0]; p_gR = p[-1]; v_gL = -v[0]; v_gR = v[-1] 

    P = PolytropicPressure(rho, gamma)
    P_gR = P[-1]; P_gL = P[0]
    F_rho[:] = rho[:]*v[:]
    F_p[:] =  rho[:]*v[:]*v[:] + P[:]
    F_rho_gR = rho_gR*v_gR; F_rho_gL = rho_gL*v_gL; 
    F_p_gR = rho_gR*v_gR*v_gR + P_gR; F_p_gL = rho_gL*v_gL*v_gL + P_gL 
   
    for j in range(1, N-1):
        rho_n[j] = 1/( 2* x[j]**alpha ) * ( rho[j+1]* x[j+1]**alpha + rho[j-1]*x[j-1]**alpha ) - dt / ( 2 * dx * x[j]**alpha ) * ( x[j+1]**alpha * F_rho[j+1] - x[j-1]**alpha *  F_rho[j-1] ) 
        p_n[j]   = 1/( 2* x[j]**alpha ) * ( p[j+1]*x[j+1]**alpha   + p[j-1]* x[j-1]**alpha   ) - dt / ( 2 * dx * x[j]**alpha ) * ( F_p[j+1] * x[j+1]**alpha  - F_p[j-1] * x[j+1]**alpha  ) + alpha/(2*x[j]**alpha) * (p[j+1]*x[j+1] + p[j-1]*x[j-1])
     
    rho_n[0]  = 1/(2* x[0]**alpha ) * (  x[1]**alpha *rho[1] + x_gL**alpha * rho_gL )  - dt / ( 2 * dx * x[0]**alpha   ) * (  x[1]**alpha * F_rho[1] -  x_gL**alpha  * F_rho_gL )
    rho_n[-1] = 1/(2* x[-1]**alpha ) * ( x_gR**alpha * rho_gR +  x[-2]**alpha *rho[-2] ) - dt / ( 2 * dx * x[-1]**alpha  ) * (  x_gR ** alpha * F_rho_gR - x[-1]**alpha * F_rho[-2] )
    p_n[0] = 1/(2* x[0]**alpha ) * (  x[1]**alpha * p[1] + x_gL**alpha * p_gL )   - dt / ( 2 * dx * x[0]**alpha)  * ( x[1]**alpha * F_p[1] - x_gL**alpha * F_p_gL ) + alpha/2 * (p[1]/x[1] + p_gL/x_gL)
    p_n[-1] = 1/(2* x[-1]**alpha ) * ( x_gR**alpha * p_gR +  x[-2]**alpha *p[-2] ) - dt / ( 2 * dx * x[-1]**alpha) * ( x_gR**alpha * F_p_gR - x[-2]**alpha * F_p[-2] ) + alpha/2 * (p_gR/x_gR + p[-2]/x[-2])
    
    rho[:] = rho_n[:]
    p[:]   = p_n[:]
    v[:]   = p_n[:] / rho_n[:]
    
    i = i + 1
    
    plt.plot(x,rho)
    plt.show()
    
    if(i%NIterationImage==0):
        plt.cla()
        plt.xlim(-1,1)
        plt.ylim(-1.,1.05)
        plt.scatter(x, rho, color='black', linewidth=0.005)
        plt.title(str(round(i*dt,3)))
        plt.pause(0.001)
    
#solution = open("solutionSodShockTube.txt", "w")
#for j in range(len(x)):
#    solution.write("%.8e %.8e %.8e \n" % (x[j], rho[j], v[j]))
#solution.close()



