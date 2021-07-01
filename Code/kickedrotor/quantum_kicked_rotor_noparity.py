"""
Author: Aditya Chincholi
This program calculates the floquet matrix for a
kicked rotor system with broken parity:

$H = p^2 / 2 + (K cos(t) + beta sin(t)^3) sum_n delta(t - n tau)$
"""

import numpy as np
import scipy.fft as fft

# Scientific Constants
K = 5
HBAR = 1
BETA = 0.1
TAU = 1

# Programming Constants
N = 1000
DIM = 2 * N + 1
FSAMPLES = 4096

def f(theta, k = K, beta = BETA, hbar = HBAR):
    """
    The function that will be fourier transformed.

    Parameters
    ----------
    theta : array_like

    Returns
    -------
    y : array_like
        y = exp(-1j * (K cos(theta) + BETA * sin^3(theta)) / HBAR)
    """
    kick = k * np.cos(theta) + beta * (np.sin(theta))**3
    return np.exp(-1j * kick / hbar)

def fourierCoeffs(func, **kwargs):
    t = np.arange(FSAMPLES) * 2 * np.pi / FSAMPLES
    samples = func(t, **kwargs)
    params = {
        "x": samples,
        "norm": "forward"
    }
    fourier_coeffs = fft.fftshift(fft.fft(**params))
    # Now we have to crop it to the correct size
    idx_zero = int(FSAMPLES/2)
    return fourier_coeffs[idx_zero-2*N:idx_zero+2*N+1]

def momentumPhase(tau, hbar):
    p = hbar * np.arange(-N, N+1)
    mm, nn = np.meshgrid(p, p, indexing="ij")
    return np.exp(-1j * nn**2 * tau / (2 * hbar))

def angularPhase(**kwargs):
    p = np.arange(-N, N+1)
    m, n = np.meshgrid(p, p, indexing="ij")
    fourier_coeffs = fourierCoeffs(f, **kwargs)
    return fourier_coeffs[m - n + 2*N]

def floquetMatrix(hbar = HBAR, tau = TAU, k = K, beta = BETA):
    return momentumPhase(hbar=hbar, tau=tau) * angularPhase(hbar=hbar, k=k, beta=beta)

if __name__ == "__main__":
    print(floquetMatrix())
