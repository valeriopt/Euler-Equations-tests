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

gamma = 1.6
err = 1e-4
errV = 2e-3
errP = err**gamma
l = 0.5

def create_video(output_video, IMAGE_DIR, fps=10):
    # Get list of images
    images = sorted([img for img in os.listdir(IMAGE_DIR) if img.endswith(".png")])
    
    if not images:
        print("No images found to create video.")
        return
    
    images.sort(key=lambda x: float(x.rsplit("/")[-1].split('.png')[0].split("T")[1]))  # Extract number and sort

    # Read the first image to determine video size
    frame = cv2.imread(os.path.join(IMAGE_DIR, images[0]))
    height, width, layers = frame.shape

    # Define the codec and initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 files
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Add each image to the video
    for image in images:
        img_path = os.path.join(IMAGE_DIR, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    # Release the video writer
    video.release()
    print(f"Video saved as {output_video}")

def create_directory(dir_name):
    try:
        os.makedirs(dir_name, exist_ok=True)
        print(f"Directory '{dir_name}' created successfully or already exists.")
    except Exception as e:
        print(f"Failed to create directory '{dir_name}': {e}")
        sys.exit(1) 

def write_arrays_to_file(filename, array1, array2):

    # Check if both arrays have the same length
    if len(array1) != len(array2):
        raise ValueError("Arrays must have the same length.")

    # Open the file in write mode
    with open(filename, 'w') as file:
        for elem1, elem2 in zip(array1, array2):
            file.write(f"{elem1}\t{elem2}\n")
    print(f"Data written to {filename} successfully.")


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
    print(peaks)

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

def fourier_transform_with_top_peaks(array, time_array, threshold=0.1, padding_factor=2, num_peaks=5):

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
    
    # Plot the magnitude spectrum and mark the top peaks
    plt.figure(figsize=(8, 6))
    plt.plot(positive_frequencies, magnitude_spectrum, label='Magnitude Spectrum')
    plt.plot(top_peak_frequencies, top_peak_magnitudes, 'ro', label='Top Peaks')
    plt.title("Magnitude Spectrum with Top Peaks")
    plt.xlabel("angular Frequency")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.savefig("Normalmodes.png")
    plt.show()
    
    return top_peak_frequencies, top_peak_periods

def fourier_transform_with_padding(array, time_array, threshold=0.1, padding_factor=2):
    
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
    frequencies = 2*np.pi* np.fft.fftfreq(n, d=dt)  # Frequencies in Hz
    
    # Only take the positive frequencies (real physical frequencies)
    positive_frequencies = frequencies[:n // 2]
    positive_coefficients = fourier_coefficients[:n // 2]
    
    # Apply threshold to filter out small frequency components (i.e., remove plateaus)
    positive_coefficients[np.abs(positive_coefficients) < threshold] = 0
    
    return positive_frequencies, positive_coefficients
    
def PlotImage( x, f, Time, energy):

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    styles = ["solid", "dashed", "dotted"]

    axes[0].cla()
    axes[1].cla()
    
    axes[0].plot(x, np.zeros(len(x)), color="black", alpha = 0.5)
    m = 0
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
    axes[1].set_xlim(0,30)
    axes[1].set_ylim(-0.15,0.15)
    
    plt.tight_layout() 
    plt.pause(0.25)  
    
def SaveImage(x, f, Time, name_image):
    styles = ["solid", "dashed", "dotted"]

    plt.plot(x, np.zeros(len(x)), color="black", alpha = 0.5)
    plt.plot(x, GetSolution(x), color = "red", alpha = 0.7)
    m = 0
    for u in f:
        if   (m == 0): LABEL = "$\u03C1/\u03C1_0$"
        elif (m == 1): LABEL = "$v \u03C1 / v_s \u03C1_0$"
        plt.plot(x, (u), linestyle= styles[m], color='black', label = LABEL)
        m = m + 1
      
    plt.xlabel('$x \u03C9/v_s$')
    plt.ylabel('Quantity')
    plt.xlim(x[0],x[-1])
    plt.ylim(-0.2,1.1)
    plt.legend()
    plt.title("T = " + str(round(Time[-1],2)))    
    plt.tight_layout() 
    plt.savefig(name_image, dpi = 200)
    plt.close()
        
def first_position_less_than(array, A):
    for index, element in enumerate(array):
        if element < A: return index
    return -1

def nearest_to_half_of_first(arr):
    if arr is None or len(arr) == 0:  # Check if the array is empty or None
        return None
    half_value = arr[0] / 2
    return min(arr, key=lambda x: abs(x - half_value)), min(range(len(arr)), key=lambda i: abs(arr[i] - half_value))

def initial(x):  return 1/2 * np.sqrt(2/np.pi) *np.exp(-x*x/2)

def analytical (x, C1):
    n = np.zeros_like(x)
    for i in range(len(x)):
        if(x[i] < np.sqrt(2.8*C1)): n[i] = (0.4 * (C1 - 1/2.8 * x[i]**2))**2.5
        else: n[i] = err
    return n

def GetSolution(x):
    C2 = 4
    C1 = 0.5
    value_integral = 1/2
    while True:
        C = (C1+C2)/2
        solution = analytical(x,C)
        integral = integrate.simpson(solution*x*x, x=x)
        if (abs(integral - value_integral) < err**2): return solution/solution[0]
        elif(integral - value_integral > err**2): C2 = C
        else: C1 = C

def Force(x, U):
    N = len(x)
    n = int(0.03 * len(x))
    i = first_position_less_than(U[0],err) + int(1. * n)

    y = 1/2 * (x-x[0]) * (x-x[0])
    
    dy = np.zeros_like(y)
    
    for idx in range(1, len(y) - 1):
        dy[idx] = (y[idx+1] - y[idx-1]) / (x[idx+1] - x[idx-1])
    dy[0] = (y[1] - y[0]) / (x[1] - x[0])
    dy[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    
    if i - n < 0 or i + n >= N:  return y, dy

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

def ForceDrag(x, U):
    
    n = int(0.05 * len(x))
    i = first_position_less_than(U[1], errV*err) + int(1.25 * n)
    
    y = np.copy(U[1])
    N = len(x)
    
    if i - n < 0 or i + n >= N:  return y

    a = 0
    y_modified = np.copy(y)
    y_modified[i+n+1:] = a
    
    x_outside = np.concatenate((x[:i-n], x[i+n+1:]))
    y_outside = np.concatenate((y_modified[:i-n], y_modified[i+n+1:]))
    
    kind = "cubic"
    interp_func = interp1d(x_outside, y_outside, kind=kind, fill_value="extrapolate")
    
    x_interp = x[i-n:i+n+1]
    y_interp = interp_func(x_interp)
    
    y_combined = np.concatenate((y[:i-n], y_interp, y_modified[i+n+1:]))        

    return y_combined

def ForcePressure(x, P): 
    
    y = np.copy(2 * P / x)
    
    N = len(x)
    n = int(0.05 * len(x))
    i = first_position_less_than(y, 0.2 * errP) + int(1.25 * n)
    
    print(x[i])
    
    if i - n < 0 or i + n >= N:  return y

    a = 0
    y_modified = np.copy(y)
    y_modified[i+n+1:] = a
    
    x_outside = np.concatenate((x[:i-n], x[i+n+1:]))
    y_outside = np.concatenate((y_modified[:i-n], y_modified[i+n+1:]))
    
    kind = "cubic"
    interp_func = interp1d(x_outside, y_outside, kind=kind, fill_value="extrapolate")
    
    x_interp = x[i-n:i+n+1]
    y_interp = interp_func(x_interp)
    
    y_combined = np.concatenate((y[:i-n], y_interp, y_modified[i+n+1:]))        

    return y_combined

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
   
def Flux_Euler(U):
    P = PolytropicPressure(U[0], gamma)
    F0 = (U[1][:])
    F1 = (U[1][:]*U[1][:]/U[0][:] + P[:])
    return [F0, F1]

def ForceSource(x,U):
    
    f = np.zeros_like(x)
    P = PolytropicPressure(U[0], gamma)
    potential, force = Force(x, U)
    for j in range(len(x)):
        f[j] = U[0][j] * force[j]  - l * U[1][j]  + 2 * P[j] / x[j]
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
            next[i][j] = 1/2 * ( x[j+1] * x[j+1] * U[i][j+1] + x[j-1] * x[j-1] * U[i][j-1] ) - dt / ( 2 * dx ) * ( x[j+1] * x[j+1] * Fluxes[i][j+1]   -  x[j-1] * x[j-1] * Fluxes[i][j-1] ) + dt /2  * (x[j+1] * x[j+1] * source[i][j+1] + x[j-1] * x[j-1] * source[i][j-1]) 
        if(i == 0) : next[i][0] = next[i][1]
        if(i == 1) : next[i][0] = 0
        next[i][-1] = next[i][-2]
    for i in range(len(U)): U[i][:] = next[i][:] / (x[:]*x[:])
   
def main(): 
    
    C = 0.305
    N = 4096
    xmax = 7
    dx = ( xmax - xmin ) / (N - 1)
    dt = C * dx
    T = 20
    NIterations = T/dt
    NIterationsImage = 100
    i = 0
    
    DIR = "integral0.5_l"+str(l)+"_gamma_" + str(gamma) +"_Q_0_C_" + str(C) + "_N_" +str(N)
    IMAGE_DIR = DIR+ "/" + "plots"
    FILE_DIR = DIR+ "/" + "files"
    create_directory(DIR)
    create_directory(IMAGE_DIR)
    create_directory(FILE_DIR) 
    
    x = np.linspace(xmin+dx, xmax+dx, N)
    rho   = np.empty_like(x)
    p     = np.empty_like(x)
    v     = np.empty_like(x)

    particle = []
    momenta = []
    Time   = []
    middle = []

    rho = InitialDistribution(x, initial)
    v   = InitialDistribution(x, zero)

    U = [rho,v]
    p[:] = v[:] * rho[:]

    k_values = np.fft.fftfreq(N, d=dx)[0:N//2] * 2 * np.pi 
    wavelenght = 2 * np.pi / k_values
    fourier = np.abs(np.fft.fft(rho)[0:N//2])

    while i < NIterations:
        CalculateNextIteration(dx, dt, x, U, Flux_Euler, Source)
        if(i%NIterationsImage == 0):
            particle.append(integrate.simpson(rho, x=x))
            momenta.append(integrate.simpson(x[10:]*x[10:]*U[0][10:], x=x[10:]))
            Time.append(i*dt)
            title= "T"+str(Time[-1])
            name_image = IMAGE_DIR+"/"+title+".png"
            name_file = FILE_DIR+"/"+title+".txt"
            value, j = nearest_to_half_of_first(U[0][25:])
            middle.append(x[j+25])
            print(round(i*dt,2), " ", momenta[-1], " ", x[j+25])
            SaveImage(x, [U[0]/U[0][25], U[1]/U[0][25]], Time, name_image)
            write_arrays_to_file(name_file, x, rho)
        i = i + 1
    write_arrays_to_file(DIR + "/" + "middle_x", Time, middle-middle[-1])
    create_video(DIR + "/" + "animation.mp4", IMAGE_DIR, 20)
    
    time,x_middle = read_arrays_from_file(DIR + "/" + "middle_x")
    x_middle_aux, time_aux  = cut_array_between_second_and_seventh_peaks(x_middle, time)
    plt.plot(time, x_middle)
    plt.plot(time_aux, x_middle_aux)
    plt.ylim(-1,1)
    plt.xlim(0,30)
    plt.show()
    positive_frequencies, periods = fourier_transform_with_top_peaks(x_middle_aux, time_aux)
    print(positive_frequencies, periods)
    
if __name__ == "__main__": main()
