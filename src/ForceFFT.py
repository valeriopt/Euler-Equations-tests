from scipy.fft import fft, ifft,  fftfreq, fftshift, ifftshift, dst, idst, rfft, irfft , rfftfreq, fht, ifht
import numpy as np
from scipy.interpolate import CubicSpline
import scipy.integrate as integrate

def GetCollectivePotential(x,n):
    
    # Gets the discretization from the arrays
    dx          = x[1]-x[0]
    N           = len(n)

    dx_2 = N/(2*N-1) * dx
    N_2 = 4 * N - 2 
    L2 = (4*N-3)/(2*N-1) * N/(N-1) * x[-1]

    # Gets an auxiliary array, which has interpolated values, and extended values for the density
    x_aux       = [k*dx_2 for k in range(0,N_2)]
    n_aux     = np.concatenate((np.interp(x_aux[0:2*N-2], x, n), [0 for k in range(2*N-2, N_2)]))
    
    #Gets the k-space discretization
    xf          = fftfreq(N_2, dx_2)[:N_2//2+1]   # Gets only the positive frequencies
    df          = xf[1]-xf[0]
    nF        = (-np.imag(rfft(n_aux*(x_aux))) * dx_2) #Fourier Transform of n (not yet spherical)
    nF[0]     = 4*np.pi*integrate.simpson(n*x*x,x=x) # Makes it spherical
    nF[1:len(xf)] = 2*np.real(nF[1:len(xf)])/(xf[1:len(xf)])

    # Gets the (spherical) Fourier Transform of the potential
    vF          = nF[0:2*N]*1
    vF[0]       = 0
    vF[1:2*N]   = nF[1:2*N]/xf[1:2*N] 

    #Inverse Fourier Transform (get potential in real space)
    xi = fftfreq(len(xf), df) [:len(xf)//2]
    V           = (-np.imag(rfft(vF))*df)[:N]
    V[1:N]      = V[1:N] / (2*np.pi*np.pi*x[1:N])
    V[0]        = 1/11 * ( 18* V[1] - 9 * V[2] + 2 * V[3] ) # Extrapolate to get potential at center

    return V    

def GetCollectiveForce(x,n):
    #Get collective force
    V = GetCollectivePotential(x,n)
    dx          = x[1]-x[0]
    F= - np.gradient(V, dx)
    F[0] = 0
    return F
