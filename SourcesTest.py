import numpy as np
from array import  *
import sys
import os
from matplotlib import pyplot as plt
import scipy.integrate as integrate
import math
import matplotlib
from scipy.integrate import odeint

from scipy.interpolate import interp1d

gamma = 1.4
err = 1e-4
errV = 1e-3
errP = err**gamma
l = 0.3

def first_position_less_than(array, A):
    for index, element in enumerate(array):
        if element < A:
            return index
    return -1

def nearest_to_half_of_first(arr):
    if arr is None or len(arr) == 0:  # Check if the array is empty or None
        return None
    half_value = arr[0] / 2
    return min(arr, key=lambda x: abs(x - half_value))

def initial(x):  return np.sqrt(2/np.pi) *np.exp(-x*x/2)

def analytical (x, C1):
    n = np.zeros_like(x)
    for i in range(len(x)):
        if(x[i] < np.sqrt(2.8*C1)): n[i] = (0.4 * (C1 - 1/2.8 * x[i]**2))**2.5
        else: n[i] = err
    return n

def GetSolution(x):
    C2 = 4
    C1 = 0.5
    while True:
        C = (C1+C2)/2
        solution = analytical(x,C)
        integral = integrate.simpson(solution, x=x)
        if (abs(integral - 1) < err**2): return solution
        elif(integral - 1 > err**2): C2 = C
        else: C1 = C

def InitialWave(x):
    solution_analytical = GetSolution(x)
    value_zero = solution_analytical[0] * 1.05
    return analytical(x, (value_zero)**(1/2.5)/0.4)
    
def Force(x, U):
    
    n = int(0.05 * len(x))
    i = first_position_less_than(U[0],err) + int(1.25 * n)
    
    y = 1/2 * x * x
    a = y[i+n]
    
    y_modified = np.copy(y)
    y_modified[i+n+1:] = a

    x_outside = np.concatenate((x[:i-n], x[i+n+1:]))
    y_outside = np.concatenate((y_modified[:i-n], y_modified[i+n+1:]))

    kind = "cubic"
    interp_func = interp1d(x_outside, y_outside, kind=kind, fill_value="extrapolate")

    x_interp = x[i-n:i+n+1]
    y_interp = interp_func(x_interp)

    y_combined = np.concatenate((y[:i-n], y_interp, y_modified[i+n+1:]))
       
    dy_combined = np.zeros_like(y_combined)

    for idx in range(1, len(y_combined) - 1):
        dy_combined[idx] = (y_combined[idx+1] - y_combined[idx-1]) / (x[idx+1] - x[idx-1])
    dy_combined[0] = (y_combined[1] - y_combined[0]) / (x[1] - x[0])
    dy_combined[-1] = (y_combined[-1] - y_combined[-2]) / (x[-1] - x[-2])
    
    return y_combined, - dy_combined

def Smoothing(U):
    
   N = len(U[0])
   x = np.linspace(0, N-1, N)
   
   y = np.copy(U[0])
   z = np.copy(U[1])
   
   i = first_position_less_than(y,err) - 10
   n = 20
   
   if i - n < 0 or i + n >= N:  return 0
           
   y_modified = np.copy(y)
   y_modified[i+n+1:] = err
   
   z_modified = np.copy(z)
   z_modified[i+n+1:] = errV * errV * err + errP + err * x[-1] + l* errV * err
   
   x_outside = np.concatenate((x[:i-n], x[i+n+1:]))
   y_outside = np.concatenate((y_modified[:i-n], y_modified[i+n+1:]))
   z_outside = np.concatenate((z_modified[:i-n], z_modified[i+n+1:]))

   kind = "cubic"
   interp_func_y = interp1d(x_outside, y_outside, kind=kind, fill_value="extrapolate")
   interp_func_z = interp1d(x_outside, z_outside, kind=kind, fill_value="extrapolate")
   
   x_interp = x[i-n:i+n+1]
   y_interp = interp_func_y(x_interp)
   z_interp = interp_func_z(x_interp)
   
   y_combined = np.concatenate((y[:i-n], y_interp, y_modified[i+n+1:]))
   z_combined = np.concatenate((z[:i-n], z_interp, z_modified[i+n+1:]))
   
   U[0][:] = y_combined[:]
   U[1][:] = z_combined[:]

def PlotImage(axes, x, f, Time, energy):
    styles = ["solid", "dashed", "dotted"]

    axes[0].cla()
    axes[1].cla()
    
    axes[0].plot(x, np.zeros(len(x)), color="black", alpha = 0.5)
    m = 0
    P = PolytropicPressure(f[0], gamma)
    for u in f:
        axes[0].plot(x, (u), linestyle= styles[m], color='black', label = str(m))
        m = m + 1
    axes[0].plot(x, GetSolution(x))        
    
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('f')
    axes[0].set_xlim(x[0],x[-1])
    axes[0].set_ylim(-0.5,1.2)
    axes[0].legend()
    axes[0].set_title(str(round(Time[-1],2)))

    m = 0
    for e in energy:
        axes[1].plot(Time, e, linestyle= styles[m], color='black')
        m = m + 1
    axes[1].set_xlabel('wavelength')
    axes[1].set_ylabel('fourier')
    axes[1].set_xlim(0,10)
    axes[1].set_ylim(-0.1,.1)
    
    plt.tight_layout() 
    plt.pause(0.25)  
        
def zero(x): return np.zeros(len(x))

def InitialDistribution(x, function): return function(x)

def PolytropicPressure(rho, gamma): 
    N = len(rho)
    x = np.linspace(0, N-1, N)
    y = np.copy(rho**gamma)
    i = first_position_less_than(y,errP)
    n = 0
    
    if i - n < 0 or i + n >= N:  return y
    
    y_modified = np.copy(y)
    y_modified[i+n+1:] = errP
    
    x_outside = np.concatenate((x[:i-n], x[i+n+1:]))
    y_outside = np.concatenate((y_modified[:i-n], y_modified[i+n+1:]))
    
    kind = "linear"
    interp_func_y = interp1d(x_outside, y_outside, kind=kind, fill_value="extrapolate")
    x_interp = x[i-n:i+n+1] 
    y_interp = interp_func_y(x_interp)
    y_combined = np.concatenate((y[:i-n], y_interp, y_modified[i+n+1:]))
    
    return y_combined
   

def Flux_Euler(u):
    P = PolytropicPressure(u[0], gamma)
    F0 = (u[1][:])
    F1 = (u[1][:]*u[1][:]/u[0][:] + P[:])
    #Smoothing([F0, F1])
    return [F0, F1]

def SourceNull(x, u): return 0

def ForceSource(x,U):
    f = np.zeros_like(x)
    potential, force = Force(x, U)
    for i in range(len(x)):
        if( U[0][i]< err ): f[i] = 0         # err * x[i]   #- l * U[1][i] / U[0][i]
        else: f[i] = U[0][i] * force[i]  - l * U[1][i]
    return f

def Source(x,U):
    source = []
    for i in range(len(U)):
        if( i == 0 ): source.append(np.zeros(len(x)))
        if( i == 1 ): source.append(ForceSource(x,U))
    return source

def CalculateNextIteration(dx, dt, x,  U, Flux, Source):
    Fluxes = Flux(U)  
    next = []
    source = Source(x,U)
    for i in range(len(U)):
        next.append(np.zeros_like(U[i]))
        for j in range(1, len(U[0])-1):
            next[i][j] = 1/2 * ( U[i][j+1] + U[i][j-1] ) - dt / ( 2 * dx ) * ( Fluxes[i][j+1]   -  Fluxes[i][j-1] ) + dt /2  * (source[i][j+1] + source[i][j-1]) 
        if(i == 0) : next[i][0] = next[i][1]
        if(i == 1) : next[i][0] = 0
        next[i][-1] = next[i][-2]
    for i in range(len(U)): U[i][:] = next[i][:]
    
C = 0.35
N = 5000
xmin = 0
xmax = 7
dx = ( xmax - xmin ) / (N - 1)
dt = C * dx
gamma = 1.4

T = 20
NIterations = T/dt
NIterationsImage = 100
i = 0

x = np.linspace(xmin, xmax, N)
rho   = np.empty_like(x)
p     = np.empty_like(x)
v     = np.empty_like(x)

particle = []
momenta = []
middle = []
Time   = []

rho = InitialDistribution(x, initial)
v   = InitialDistribution(x, zero)

A = input(" ")

U = [rho,v]
p[:] = v[:] * rho[:]

k_values = np.fft.fftfreq(N, d=dx)[0:N//2] * 2 * np.pi 
wavelenght = 2 * np.pi / k_values
fourier = np.abs(np.fft.fft(rho)[0:N//2])

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

while i < NIterations:
    CalculateNextIteration(dx, dt, x, U, Flux_Euler, Source)
    if(i%NIterationsImage == 0):
        particle.append(integrate.simpson(rho, x=x))
        momenta.append(integrate.simpson(U[1], x=x))
        middle.append(nearest_to_half_of_first(rho))
        Time.append(i*dt)
        fourier = np.abs(np.fft.fft(rho)[0:N//2])
        PlotImage(axes, x, [U[0], np.abs(U[1]/U[0]), U[1]],  Time, [middle-middle[0]])
    i = i + 1
