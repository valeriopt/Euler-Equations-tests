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
    u = np.zeros_like(x)
    f1 = 1
    f2 = np.linspace(2, 5, 2)
    for j in range(len(u)):
        for f in range(len(f2)): 
            if (0.25<x[j]<.55): u[j] = u[j] +  np.sin(2*np.pi*f1*(x[j]-1))*np.sin(2*np.pi*f2[f]*x[j])
            else: u[j] = 0
    #u = -np.arctan(x-2)
    return 1+ 0.2 * u

def PlotImage(axes, x, f, Time, energy):
    styles = ["solid", "dashed", "dotted"]

    axes[0].cla()
    axes[1].cla()
    
    m = 0
    for u in f:
        axes[0].plot(x, u, linestyle= styles[m], color='black', label = str(m))
        m = m + 1
        
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('f')
    axes[0].set_xlim(x[0],x[-1])
    axes[0].set_ylim(-0.1,1.1)
    axes[0].legend()
    axes[0].set_title(str(round(Time[-1],2)))

    m = 0
    for e in energy:
        axes[1].plot(Time, e, linestyle= styles[m], color='black')
        m = m + 1
    axes[1].set_xlabel('wavelength')
    axes[1].set_ylabel('fourier')
    axes[1].set_xlim(0,5)
    axes[1].set_ylim(0,25)
    
    plt.tight_layout()  # Ensure spacing between plots
    plt.pause(0.25)  # Pause to view each figure
        
def zero(x):
    return np.zeros(len(x))

def InitialDistribution(x, function):
    return function(x)

def PolytropicPressure(rho, gamma):
    return rho**gamma

def PolytropicPressureEnergy(rho, v, E, gamma):
    return (gamma-1) * (E - 1/2 * rho * v * v) 

def Flux_Euler(u):
    P = PolytropicPressure(u[0], gamma)
    F0 = (u[1][:])
    F1 = (u[1][:]*u[1][:]/u[0][:] + P[:])
    return [F0, F1]

def SourceNull(x, u):
    return 0

def Source(x,U):
    source = []
    for i in range(len(U)):
        if( i == 0 ): source.append(np.zeros(len(x)))
        if( i == 1 ): source.append(x)
    return source

def CalculateNextIteration(dx, dt, x,  U, Flux, Source):
    Fluxes = Flux(U)        
    next = []
    source = Source(x,U)
    for i in range(len(U)):
        next.append(np.zeros_like(U[i]))
        for j in range(1, len(U[0])-1):
            next[i][j] = 1/2 * ( U[i][j+1] + U[i][j-1] ) - dt / ( 2 * dx ) * ( Fluxes[i][j+1]   -  Fluxes[i][j-1] ) + dt /2  * (source[i][j+1] + source[i][j-1]) 
        next[i][0] = next[i][1]
        next[i][-1] = next[i][-2]
    for i in range(len(U)): U[i][:] = next[i][:]
    
C = 0.1
N = 1000
xmin = 0
xmax = 1
dx = ( xmax - xmin ) / (N - 1)
dt = C * dx
gamma = 1.4

T = 1
NIterations = T/dt
NIterationsImage = 20
i = 0

x = np.linspace(xmin, xmax, N)
rho   = np.empty_like(x)
p     = np.empty_like(x)
v     = np.empty_like(x)

particle = []
Time   = []

rho = InitialDistribution(x,initial)
v   = InitialDistribution(x,zero)
plt.plot(x,rho)
plt.show()
U = [rho,v]
p[:] = v[:] * rho[:]

k_values = np.fft.fftfreq(N, d=dx)[0:N//2] * 2 * np.pi 
wavelenght = 2 * np.pi / k_values
fourier = np.abs(np.fft.fft(rho)[0:N//2])

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

while i < NIterations:
    CalculateNextIteration(dx, dt, x, U, Flux_Euler, Source)
    if(i%NIterationsImage == 0):
        particle.append(integrate.simpson(rho*x*x, x=x))
        Time.append(i*dt)
        fourier = np.abs(np.fft.fft(rho)[0:N//2])
        PlotImage(axes, x, [U[0], v],  Time, [particle])
    i = i + 1