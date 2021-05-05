"""
Author: Aditya Chincholi
Purpose: To calculate time evolution of entanglement entropy
        of the theta1 and theta2+theta3 momentum spaces in the
        quasiperiodic kicked rotor. We consider the Hamiltonian
        H = p1^2/2 + p2*ω2 + p3*ω3 + K cosθ1 (1 + α cosθ2 cosθ3) Σ δ(t - n)
Decisions: We use the momentum basis for all three dimensions. This is because
        even though the fourier transform is analytically solvable, the
        resultant expression contains two dirac delta terms.
"""

import numpy as np
import scipy.fft as fft
from scipy.special import xlogy
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse import eye as sparse_eye
import matplotlib.pyplot as plt
from cython.parallel import prange
cimport cython

from libc.math cimport sqrt

cdef extern from "complex.h":
    double cos(double x) nogil
    double sin(double x) nogil
    float complex I

cdef extern from "math.h":
    float M_PI

# Scientific Constants
cdef float HBAR = 2.89
cdef float K = 5
cdef float ALPHA = 0.2
cdef float OMEGA2 = 2 * M_PI * sqrt(5)
cdef float OMEGA3 = 2 * M_PI * sqrt(13)

# Program Constants
cdef int N = 3
cdef int DIM = 2 * N + 1
cdef float EPSILON = 1e-6
cdef int TIMESTEPS = 5
cdef FSAMPLES = 32

def printMatrix(matrix, name):
    """
    Prints the given matrix prettily. For debugging purposes.
    """
    print(f"{name}:")
    print("-"*len(name))
    print(matrix)
    print()


def kickFunction(theta1, theta2, theta3):
    """
    Returns the kick part of floquet operator.
    """
    # quasikick = (1 + ALPHA * np.cos(theta2) * np.cos(theta3))
    # return np.exp(-1j * K * np.cos(theta1) * quasikick / HBAR)
    return np.zeros(theta1.shape)

def getInitialState():
    """
    Returns the initial state as tensor product of the 3 momentum spaces.
    """
    state1 = np.zeros(DIM)
    state1[N] = 1 # State |0>

    state2 = np.ones(DIM) / np.sqrt(DIM)
    state23 = np.kron(state2, state2)
    return np.kron(state1, state23)

def getInitialDensity():
    """
    Returns the density matrix of the initial state.
    """
    state = getInitialState()
    return np.outer(state, state)

def getKickFourierCoeffs(func):
    """
    Returns the 3d fourier coefficient matrix of the function kick(t1, t2, t3).
    """
    theta = np.arange(FSAMPLES) * 2 * np.pi / (FSAMPLES)
    theta1, theta2, theta3 = np.meshgrid(theta, theta, theta)
    x = func(theta1, theta2, theta3)
    y = fft.fftshift(fft.fftn(x, norm="forward"))
    return y.astype(np.complex64)

cdef float complex complex_exp(float angle) nogil:
    return cos(angle) + I * sin(angle)

@cython.boundscheck(False)
@cython.wraparound(False)
def getDenseFloquetOperator(float complex[:, :, :] fourier_coeffs not None):
    """
    Returns the dense version of the floquet operator.
    """

    F = np.zeros((DIM**3, DIM**3), dtype=np.complex64)
    cdef float complex[:, :] F_view = F

    cdef int m1, m2, m3, n1, n2, n3
    cdef int shift = (FSAMPLES / 2) + 1
    cdef int row, col
    m1 = m2 = m3 = n1 = n2 = n3 = row = col = 0
    cdef float angle = 0.0
    cdef float complex fourier = 0 + 0*I

    for m1 in prange(-N, N+1, nogil=True):
        for m2 in prange(-N, N+1):
            for m3 in prange(-N, N+1):
                for n1 in prange(-N, N+1):
                    for n2 in prange(-N, N+1):
                        for n3 in prange(-N, N+1):
                            angle = (HBAR / 2) * n1**2 + n2 * OMEGA2 + n3 * OMEGA3
                            fourier = fourier_coeffs[m1-n1+shift, m2-n2+shift, m3-n3+shift]
                            row = ((m1 + N) * (DIM**2)) + ((m2 + N) * DIM) + (m3 + N)
                            col = ((n1 + N) * (DIM**2)) + ((m3 + N) * DIM) + (n3 + N)
                            F_view[row, col] = complex_exp(-angle) * fourier

    return F

def getFloquetOperator():
    """
    Returns the floquet operator for the 3d quasiperiodic kicked rotor.
    """
    fourier_coeffs = getKickFourierCoeffs(kickFunction)

    F = getDenseFloquetOperator(fourier_coeffs)
    sign, logdet = np.linalg.slogdet(F)
    print(f"slogdet(F) = {sign} * exp({logdet})")
    Fh = np.conjugate(F.T)
    # with np.printoptions(precision=4, threshold=np.inf, suppress=True, linewidth=120):
        # printMatrix(F.dot(Fh), "F x Fh")
    F[np.abs(F) < EPSILON] = 0
    return csr_matrix(F), csr_matrix(Fh)

def evolve(rho, F, Fh):
    """
    Evolves the density matrix by one unit time step.
    """
    return csr_matrix.dot(F.dot(rho), Fh)

def partialTrace(rho):
    """
    Returns partial trace over theta2+theta3 space.
    """
    rho_product = rho.reshape((DIM,)*6)
    return np.einsum("ijkljk", rho_product) # Einstein Summation Convention

def getEntanglementEntropy(rho1):
    """
    Returns the von Neumann entropy of the given density matrix.
    """
    eigvals = np.linalg.eigvals(rho1)
    return np.sum(-xlogy(eigvals, eigvals))

def plotEntropies(entropies):
    """
    Plots the entanglement entropies as a function of time.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1)

    t = np.arange(1, TIMESTEPS+1)
    ax.semilogy(t, entropies, marker="o")
    ax.set_xlabel("Time (t)")
    ax.set_ylabel(r"Entanglement Entropy ($\sigma$)")
    plt.savefig("plots/entanglement_entropy.png")

def main():
    """
    The main function which handles the entire execution sequence.
    """
    print("Starting to compute F...")
    F, Fh = getFloquetOperator() # F is sparse CSR, Fh is sparse CSC
    # print(f"F has {F.count_nonzero()} non-zero elements out of {DIM**6}")
    print("Floquet operator computation over. Density operator starting...")
    rho = getInitialDensity()
    rho1 = partialTrace(rho)
    print("Trace of rho:", np.trace(rho))
    print("Trace of rho1:", np.trace(rho1))
    print("von Neumann Entropy:", getEntanglementEntropy(rho1))

    entropies = np.empty(TIMESTEPS)
    print("Starting evolution...")
    for t in range(TIMESTEPS):
        print(f"Iteration {t} starting...")
        rho1 = partialTrace(rho)
        print(f"Trace of rho1: {np.trace(rho1)}")
        entropy = getEntanglementEntropy(rho1)
        print(f"Calculated von Neumann entropy is: {entropy}")
        entropies[t] = entropy
        rho = evolve(rho, F, Fh)
        print()

    plotEntropies(entropies)

if __name__ == "__main__":
    main()
    plt.show()
