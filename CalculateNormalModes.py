import numpy as np
from array import  *
import sys, os
from matplotlib import pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import cv2
import matplotlib

params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)
matplotlib.rcParams.update({'font.size': 15})

def find_closest_value(array, value):
    closest = min(array, key=lambda x: abs(x - value))
    return closest

def Breathing_Modes(n, gamma):
    return np.sqrt(2*n + (gamma-1) * (1 + 2 * n ** 2))

def read_arrays_from_file(filename):

    array1 = []
    array2 = []

    # Open the file in read mode
    with open(filename, 'r') as file:
        for line in file:
            # Split the line into two parts (assuming tab-separated values)
            parts = line.strip().split('\t')
            if len(parts) == 2:  # Ensure there are two parts
                # Convert parts to float and add to respective arrays
                array1.append(float(parts[0]))
                array2.append(float(parts[1]))

    print(f"Data read from {filename} successfully.")
    return array1, array2

def cut_array_between_second_and_seventh_peaks(array, x):
    
    if len(array) < 3:
        raise ValueError("The array must have at least 3 elements to detect peaks.")
    if len(array) != len(x):
        raise ValueError("The array and x must have the same length.")

    # Use scipy to find peaks
    peaks, _ = find_peaks(array)

    # Ensure there are at least 7 peaks
    if len(peaks) < 7:
        raise ValueError("The array must have at least 7 peaks to perform this operation.")

    # Get the indices of the second and seventh peaks
    second_peak_index = peaks[1]  # Second peak
    seventh_peak_index = peaks[7]  # Seventh peak

    # Slice the array and x between these peaks
    start = second_peak_index
    end = seventh_peak_index

    return array[start:end + 1], x[start:end + 1]

def fourier_transform_with_top_peaks(array, time_array, threshold=0.1, padding_factor=2):

    # Ensure the time array and data array have the same length
    if len(array) != len(time_array):
        raise ValueError("The time array and data array must have the same length.")
    
    # Calculate the sampling interval (dt) from the time array
    dt = time_array[1] - time_array[0]  # Assuming uniform sampling intervals
    
    # Perform zero padding by increasing the length of the signal
    padded_length = len(array) * padding_factor  # New length after padding
    padded_array = np.pad(array, (0, padded_length - len(array)), 'constant')
    
    # Perform the Fourier Transform (FFT) on the padded signal
    fourier_coefficients = np.fft.fft(padded_array)
    
    # Compute the frequencies corresponding to the Fourier coefficients
    n = len(padded_array)  # New length after padding
    sampling_rate = 1 / dt  # Sampling rate is the inverse of the time interval
    frequencies = np.fft.fftfreq(n, d=dt)  # Frequencies in Hz
    
    # Only take the positive frequencies (real physical frequencies)
    positive_frequencies = 2*np.pi * frequencies[:n // 2]
    positive_coefficients = fourier_coefficients[:n // 2]
    
    # Apply threshold to filter out small frequency components (i.e., remove plateaus)
    positive_coefficients[np.abs(positive_coefficients) < threshold] = 0
    
    # Find the peaks in the magnitude of the Fourier coefficients (significant frequencies)
    magnitude_spectrum = np.abs(positive_coefficients)
    
    return positive_frequencies, magnitude_spectrum


def Get_Normal_Modes(positive_frequencies, magnitude_spectrum, num_peaks = 5):

    peaks, _ = find_peaks(magnitude_spectrum)  # Get indices of peaks in the magnitude spectrum
    
    # Get the corresponding frequencies and magnitudes for the peaks
    peak_frequencies = positive_frequencies[peaks]
    peak_magnitudes = magnitude_spectrum[peaks]
    
    # Sort peaks by magnitude (descending order) and get the top 'num_peaks' peaks
    sorted_indices = np.argsort(peak_magnitudes)[::-1]  # Sort by descending magnitude
    top_peaks_indices = sorted_indices[:num_peaks]  # Select the top 'num_peaks'
    
    # Get the frequencies and magnitudes of the top peaks
    top_peak_frequencies = peak_frequencies[top_peaks_indices]
    top_peak_magnitudes = peak_magnitudes[top_peaks_indices]
    
    # Calculate the periods for each of the top peaks
    top_peak_periods = 2*np.pi / top_peak_frequencies  # Period = 1 / Frequency
    
    return top_peak_frequencies, top_peak_magnitudes, top_peak_periods


def main(): 
    
    gamma = 1.375
    n = [1, 2, 3, 4]
    normal_modes_analytical = []
    l = 0.5
    integral = [0.5]
    
    plt.figure(figsize=(8, 6))
    for i in n: 
        breathing_mode = Breathing_Modes(i, gamma)
        normal_modes_analytical.append(breathing_mode)
        plt.plot([breathing_mode,breathing_mode], [0,11], alpha = 0.5, color = "red")
        plt.text(breathing_mode + 0.1, 9.5, "$n= $" + str(i), fontsize=10, color='black')
    for i in integral:
        DIR = "integral" +str(i)+ "_l"+str(l)+"_gamma_" + str(gamma) +"_Q_0" + "_C_" + "0.32"# + "_N" + "_5000"
        print(DIR)
        time,x_middle = read_arrays_from_file(DIR + "/" + "middle_x")
        x_middle_aux, time_aux  = cut_array_between_second_and_seventh_peaks(x_middle, time)
        frequencies, magnitude_spectrum = fourier_transform_with_top_peaks(x_middle_aux, time_aux)
        top_peak_frequencys, top_peaks_magnitudes, periods = Get_Normal_Modes(frequencies, magnitude_spectrum)
        plt.plot(frequencies, magnitude_spectrum, label='$N = 4\u03C0$ ' + str(i) )
        plt.plot(top_peak_frequencys, top_peaks_magnitudes, 'ro')
    plt.title("Breathing modes of vibration")
    plt.xlabel("$\u03C9/\u03C9_0$")
    plt.ylabel("$F(\u03C1_{1/2}/\u03C1_0)$")
    plt.legend()
    plt.grid(True)
    plt.xlim(0,10)
    plt.ylim(2,10)
    plt.savefig("BreathingModesChangingNumberOfParticles" + str(gamma)+ ".png", dpi = 300)
    plt.show()
    
    gamma = [1.2, 1.25, 1.3, 1.35, 1.375, 1.4, 1.45, 1.5, 1.55, 1.6]
    n = [1,2,3]
    wB = []
        
    plt.figure(figsize=(8, 6))
    G = np.linspace(1, 1.7)
    styles = ["solid", "dashed", "dotted"]
    colors = ["darkred", "darkblue", "darkgreen"]
    
    for i in range(len(n)): 
        wB.append([]*len(gamma))
        for j in range(len(gamma)): wB[i].append(0)
        plt.plot(G, Breathing_Modes(n[i], G), color = "black", linestyle = styles[i], label= "n= " + str(n[i]))
    for i in range(len(gamma)):
        if(gamma[i] == 1.6 or gamma[i] == 1.55) : C = 0.305
        elif(gamma[i] == 1.375): C = 0.32
        else: C = 0.31 
        DIR = "integral" +str(integral[0])+ "_l"+str(l)+"_gamma_" + str(gamma[i]) +"_Q_0" + "_C_" + str(C) 
        time,x_middle = read_arrays_from_file(DIR + "/" + "middle_x")
        x_middle_aux, time_aux  = cut_array_between_second_and_seventh_peaks(x_middle, time)
        frequencies, magnitude_spectrum = fourier_transform_with_top_peaks(x_middle_aux, time_aux)
        top_peak_frequencys, top_peaks_magnitudes, periods = Get_Normal_Modes(frequencies, magnitude_spectrum)
        for j in range(len(n)): 
            wB[j][i] = find_closest_value(top_peak_frequencys, Breathing_Modes(n[j], gamma[i]))
            if(wB[j][i] == wB[j-1][i]): wB[j][i] = 0 
    for i in range(len(n)):
        plt.plot(gamma, wB[i], 'o', color = colors[i])
    plt.xlabel("$\u03B3$")
    plt.ylabel("$\u03C9_B$") 
    plt.ylim(1,4.5)
    plt.xlim(1.1,1.7)
    plt.title("Breathing modes (for different $\u03B3$)")
    plt.savefig("BreathingModesChangingNumberOfParticles" + str(gamma)+ ".png", dpi = 300)
    plt.grid(True)
    plt.legend()
    plt.savefig("BreathingModesChangingPolytropicGamma.png", dpi = 300)
    plt.show()
        
if __name__ == "__main__": main()