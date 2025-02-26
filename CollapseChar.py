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
    return np.sqrt( (gamma-1) * (n ) * ( 2 * n + 3 ) )

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

def cut_array_between_specified_peaks(array, x, peak1_num, peak2_num):

    if len(array) < 3:
        raise ValueError("The array must have at least 3 elements to detect peaks.")
    if len(array) != len(x):
        raise ValueError("The array and x must have the same length.")
    if peak1_num <= 0 or peak2_num <= 0:
        raise ValueError("Peak numbers must be positive integers.")
    if peak1_num >= peak2_num:
        raise ValueError("The first peak number must be less than the second peak number.")

    # Use scipy to find peaks
    peaks, _ = find_peaks(array)

    # Ensure there are enough peaks
    if len(peaks) < peak2_num:
        raise ValueError(f"The array must have at least {peak2_num} peaks to perform this operation.")

    # Get the indices of the specified peaks
    peak1_index = peaks[peak1_num - 1]  # Convert ordinal to zero-based index
    peak2_index = peaks[peak2_num - 1]  # Convert ordinal to zero-based index

    # Slice the array and x between these peaks
    start = peak1_index
    end = peak2_index

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
    
    gamma = 1.01
    n = [1, 2, 3]
    normal_modes_analytical = []
    l = 0.3
    integral = [0.5]
    breathing_mode_wp = np.sqrt(3)
    
    Q = 0
    plt.figure(figsize=(8, 6))
 
    QInitial  = 125
    C = 0.14
    DIR = "integral0.5_l"+str(l)+"_gamma_" + str(gamma) +"_Q_" + str(Q) + "_C_" + str(C) + "collapse" + str(QInitial)

    #DIR = "integral" +str(0.5)+ "_l"+str(l)+"_gamma_" + str(gamma) +"_Q_" + str(Q) + "_C_" + "0.14" + "collapse_2" # "_shockwave" #+ "_aux" # + "_N" + "_5000"
    time,x_middle = read_arrays_from_file(DIR + "/" + "radius.txt")
    print(time)
    print(x_middle)
    time,x_middle2 = read_arrays_from_file(DIR + "/" + "centraldensity.txt")
    #x_middle_aux, time_aux  = cut_array_between_specified_peaks(x_middle, time, 2, 7)
    
    plt.plot(time, x_middle)
    plt.plot(time, x_middle2)
    #plt.plot(time_aux, x_middle_aux)
    #plt.ylim(-0.2,0.2)
    plt.show()
    


if __name__ == "__main__": main()
