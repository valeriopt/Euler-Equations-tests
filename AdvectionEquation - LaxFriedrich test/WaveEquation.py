import numpy as np
from array import  *
import sys
import os
from matplotlib import pyplot as plt
import scipy.integrate as integrate
import math
import matplotlib
from scipy.integrate import odeint
from scipy.fft import *
import time


styles = ["solid", "dashed", "dotted"]

def initial(x):
    s = 0.1
    return np.exp(-(x-1)*(x-1)/(s*s))

def initial_s(x):
    s = np.zeros_like(x)
    return s


def derivative_X(x, f):
    der = np.empty_like(f)
    dx = x[1] - x[0]
    for j in range(0,len(f)-2): der[j] = (1*f[j-2]-8*f[j-1]+0*f[j+0]+8*f[j+1]-1*f[j+2])/(12*1.0*dx**1)
    der[-2] = (1*f[-4]-8*f[-3]+0*f[-2]+8*f[-1]-1*f[0])/(12*1.0*dx**1)
    der[-1] = (1*f[-3]-8*f[-2]+0*f[-1]+8*f[0]-1*f[1])/(12*1.0*dx**1)
    return der

def PlotImage(axes, x, f, Time, energy):
    sg = 0.1
    v = 1
    axes[0].cla()
    axes[1].cla()
    
    m = 0
    for u in f:
        axes[0].plot(x, u, linestyle= styles[m], color='black')
        m = m + 1
        
    axes[0].plot(x, np.exp(-(x-1-v*Time[-1])*(x-1-v*Time[-1])/(sg*sg)), color = "red")
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('f')
    axes[0].set_xlim(x[0],x[-1])
    axes[0].set_ylim(0,1.05)

    m = 0
    for e in energy:
        axes[1].plot(Time, e, linestyle= styles[m], color='black')
        m = m + 1
    axes[1].set_xlabel('wavelenght')
    axes[1].set_ylabel('fourier')
    axes[1].set_xlim(0,5)
    #axes[1].set_ylim(0,2)

    #plt.tight_layout()# Draw updated line
    plt.tight_layout()  # Ensure spacing between plots
    plt.pause(0.25)  # Pause to view each figure
        
f = 2
n = 3
C0 = 1
C = np.zeros(n)
for j in range(n):
    C[j] = C0 / f**j

N = 1000
xmin = 0
xmax = 4
dx = ( xmax - xmin ) / (N)
dt = C * dx
a0 = 1

v = 1

NIterations = np.zeros(n)
NIterations0 = 10000
for j in range(n):
    dt[j] = C[j] * dx
    NIterations[j] = int(NIterations0*f**j)

x = np.array([xmin + dx*j for j in range(N)])

r     = []
s     = []
u     = []

r_n   = []
s_n   = []

F_r = []
F_s = []

fourier = []

energy = []
Time   = []

sg = 0.1

for j in range(n):
    r.append(v * derivative_X(x,initial(x)))
    s.append(- v * derivative_X(x,initial(x)))
    u.append(initial(x))
    
    r_n.append(np.empty_like(x))
    s_n.append(np.empty_like(x))
    
    F_r   .append(np.empty_like(x))
    F_s   .append(np.empty_like(x))
    
    energy.append([])
    
    fourier.append(np.fft.fft(u))


k_values = np.fft.fftfreq(N , d=dx)[0:N//2] * 2 * np.pi 
wavelenght = 2 * np.pi / k_values

NIterationsImages = 5
i = 0

fig, axes = plt.subplots(1, 2, figsize=(12, 6))  #

while i < NIterations[-1]:
    
    if(i%f**(n-1) == 0  and i%NIterationsImages == 0):  
        Time.append(i*dt[-1])
        for k in range(n):  
            energy[k].append(np.log10(sum(u[k]*u[k])))
            fourier[k] = np.abs(np.fft.fft(u[k])[0:N//2])
        PlotImage(axes, x, u, wavelenght, fourier)
    
    for k in range(n): 
        if( i%f**(n-k) ==0 ):
            
            F_r[k][:] = - v * s[k]
            F_s[k][:] = - v * r[k] 
    
            for j in range(1, N-1):
                r_n[k][j] = 1/2 * ( r[k][j+1] + r[k][j-1] ) - dt[k] / ( 2 * dx ) * ( F_r[k][j+1] - F_r[k][j-1] )
                s_n[k][j] = 1/2 * ( s[k][j+1] + s[k][j-1] ) - dt[k] / ( 2 * dx ) * ( F_s[k][j+1] - F_s[k][j-1] )
            
            u[k] = u[k] + dt[k] * s[k]
            
            r_gR = r_n[k][1]
            r_gL = r_n[k][-2]
            
            s_gR = s_n[k][1]
            s_gL = s_n[k][-2]
            
            r_n[k][0] = r_gL
            s_n[k][0] = s_gL
            
            r_n[k][-1] = r_gR
            s_n[k][-1] = s_gR
            
            r[k][:] = r_n[k][:]
            s[k][:] = s_n[k][:]    
    i = i + 1
    
    
        
