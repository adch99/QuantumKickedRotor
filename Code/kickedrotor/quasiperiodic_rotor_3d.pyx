"""
Author: Aditya Chincholi
Purpose: Cython code for calculating the floquet operator for
        the 3d kicked rotor.
"""
import numpy as np
import scipy.fft as fft

from cython.parallel import prange
cimport cython

from libc.math cimport sqrt, cos, sin, isnan
from libc.stdio cimport printf

cdef extern from "math.h":
    double M_PI

import kickedrotor.params as params

# Scientific Constants
cdef double HBAR = params.HBAR
cdef double K = params.K
cdef double ALPHA = params.ALPHA
cdef double OMEGA2 = params.OMEGA2
cdef double OMEGA3 = params.OMEGA3
cdef double complex I = 1j

# Program Constants
cdef int N = params.N
cdef int DIM = 2 * N + 1
cdef FSAMPLES = params.FSAMPLES
DTYPE = params.DTYPE

def kickFunction(theta1, theta2, theta3):
    """
    Returns the kick part of floquet operator.
    """
    quasikick = (1 + ALPHA * np.cos(theta2) * np.cos(theta3))
    return np.exp(-1j * K * np.cos(theta1) * quasikick / HBAR)


def getKickFourierCoeffs(func):
    """
    Returns the 3d fourier coefficient matrix of the function kick(t1, t2, t3).
    """
    theta = np.arange(FSAMPLES) * 2 * np.pi / (FSAMPLES)
    theta1, theta2, theta3 = np.meshgrid(theta, theta, theta, indexing="ij")
    x = func(theta1, theta2, theta3)
    y = fft.fftshift(fft.fftn(x, norm="forward"))
    return y.astype(DTYPE)

cdef double complex complex_exp(double angle) nogil:
    return <double> cos(angle) + I * <double> sin(angle)

@cython.boundscheck(False)
@cython.wraparound(False)
def getDenseFloquetOperator(fourier_coeffs):
    """
    Returns the dense version of the floquet operator.
    """

    F = np.zeros((DIM**3, DIM**3), dtype=DTYPE)
    cdef double complex[:, :] F_view = F
    cdef double complex[:, :, :] fourier_view = fourier_coeffs

    cdef int m1, m2, m3, n1, n2, n3
    cdef int shift = FSAMPLES / 2
    cdef int row, col
    # m1 = m2 = m3 = n1 = n2 = n3 = row = col = 0
    cdef double angle = 0.0
    cdef double complex fourier = 0 + 0*I
    cdef double complex phase

    cdef int DIMSQ = DIM**2

    for m1 in prange(-N, N+1, nogil=True):
        for m2 in prange(-N, N+1):
            for m3 in prange(-N, N+1):
                for n1 in prange(-N, N+1):
                    for n2 in prange(-N, N+1):
                        for n3 in prange(-N, N+1):
                            angle = (HBAR / 2) * n1**2 + n2 * OMEGA2 + n3 * OMEGA3
                            phase = complex_exp(-angle)
                            fourier = fourier_view[m1-n1+shift, m2-n2+shift, m3-n3+shift]
                            row = (m1 + N) * DIMSQ + (m2 + N) * DIM + (m3 + N)
                            col = (n1 + N) * DIMSQ + (n2 + N) * DIM + (n3 + N)
                            F_view[row, col] = phase * fourier
                            # F_view[row, col] = fourier

    return F

def getFloquetOperator():
    fourier_coeffs = getKickFourierCoeffs(kickFunction)
    F = getDenseFloquetOperator(fourier_coeffs)
    return F
