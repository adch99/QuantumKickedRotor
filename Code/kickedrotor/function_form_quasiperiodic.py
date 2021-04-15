
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.special import jv
import matplotlib.pyplot as plt

N = 1000
K = 5

def target(theta1, theta2, theta3):
    return np.exp(-1j * K * np.cos(theta1)\
        * (1 + epsilon * np.cos(theta2) * np.cos(theta3)))

def single(theta):
    return np.exp(-1j * K * np.cos(theta))

def main():
    theta = np.linspace(0, 2*np.pi, N)
    spacing = 2*np.pi / N
    y = single(theta)
    fourier_coeffs = fft(y)
    freqs = fftfreq(N, d=spacing)
    
    # plt.semilogy(freqs, np.abs(fourier_coeffs), marker="o")
    # plt.xlabel("Frequency")
    # plt.ylabel("Amplitude")

if __name__ == "__main__":
    main()
    plt.show()
