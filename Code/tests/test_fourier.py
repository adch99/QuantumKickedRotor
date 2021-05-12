"""
Author: Aditya Chincholi
Purpose: To verify that our code for the fourier transform
        and the floquet operator works.
"""

import numpy as np
import scipy.fft as fft
from scipy.integrate import nquad

# ALPHA = 0.5
#
# def f_real(t1, t2, t3, m1, m2, m3):
#     # kick = -np.cos(t1) * (1 + ALPHA * np.cos(t2) * np.cos(t3))
#     kernel = -(m1*t1 + m2*t2 + m3*t3)
#     # return np.cos(kernel + kick)
#     return np.cos(kernel + t1)
#
# def f_imag(t1, t2, t3, m1, m2, m3):
#     # kick = -np.cos(t1) * (1 + ALPHA * np.cos(t2) * np.cos(t3))
#     kernel = -(m1*t1 + m2*t2 + m3*t3)
#     # return np.sin(kernel + kick)
#     return np.sin(kernel + t1)
#
# def f_kick_full(t1, t2, t3):
#     # kick = -np.cos(t1) * (1 + ALPHA * np.cos(t2) * np.cos(t3))
#     # return np.exp(1j * kick)
#     return np.exp(1j * t1)
#


def f(t1, t2, t3):
    kick = -np.cos(t1) * (1 + 0.5 * np.cos(t2) * np.cos(t3))
    return np.cos(kick) + 1j * np.sin(kick)
    # return np.exp(-(t1**2 + t2**2) / 2)

def g_real(t1, t2, t3, m1, m2, m3):
    kick = -np.cos(t1) * (1 + 0.5 * np.cos(t2) * np.cos(t3))
    kernel = -m1*t1 - m2*t2 - m3*t3
    return np.cos(kick + kernel)
    # return f(t1, t2) * np.cos(kernel)

def g_imag(t1, t2, t3, m1, m2, m3):
    kick = -np.cos(t1) * (1 + 0.5 * np.cos(t2) * np.cos(t3))
    # kick = -(t1**2 + t2**2) / 2
    kernel = -m1*t1 - m2*t2 - m3*t3
    # return f(t1, t2) * np.sin(kernel)
    return np.sin(kick + kernel)

def findIntegral(m1, m2, m3):
    params = (m1, m2, m3)
    ranges = [(0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi)]
    result_real, abserr_real = nquad(g_real, ranges, args=params)
    result_imag, abserr_imag = nquad(g_imag, ranges, args=params)
    result = (result_real + 1j * result_imag) / (2*np.pi)**3
    abserr = (abserr_real + 1j * abserr_imag) / (2*np.pi)**3
    return result, abserr

def getFourier():
    theta = np.arange(64) * 2 * np.pi / 64
    t1, t2, t3 = np.meshgrid(theta, theta, theta)
    x = f(t1, t2, t3)
    y = fft.fftshift(fft.fftn(x, norm="forward"))
    return y

def main():
    m1, m2, m3 = (0, 0, 0)
    fourier_coeffs = getFourier().T
    fourier = fourier_coeffs[m3+32, m1+32, m2+32]
    integral, abserr = findIntegral(m1, m2, m3)
    print(f"From Fourier: {fourier}")
    print(f"From Integral: {integral}")
    print("Difference:", np.abs(fourier - integral))
    # print("(2 pi)**3 =", (2*np.pi)**3)

def plotFourier(fourier_coeffs):
    plt.imshow(fourier_coeffs.real)
    plt.title("Real")
    plt.xlabel("m1")
    plt.ylabel("m2")

if __name__ == "__main__":
    main()
