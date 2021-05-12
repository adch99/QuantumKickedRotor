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
from scipy.special import xlogy, seterr
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse import eye as sparse_eye
from scipy.linalg import eigvals
import matplotlib.pyplot as plt
from cython.parallel import prange
cimport cython

from libc.math cimport sqrt, cos, sin, isnan
from libc.stdio cimport printf

cdef extern from "math.h":
    double M_PI

# Scientific Constants
cdef double HBAR = 2.89
cdef double K = 5
cdef double ALPHA = 0.2
cdef double OMEGA2 = 2 * M_PI * sqrt(5)
cdef double OMEGA3 = 2 * M_PI * sqrt(13)
cdef double complex I = 1j

# Program Constants
cdef int N = 10
cdef int DIM = 2 * N + 1
cdef double EPSILON = 1e-6
cdef int TIMESTEPS = 20
cdef FSAMPLES = 32
DTYPE = np.complex128

def printMatrix(matrix, name):
    """
    Prints the given matrix prettily. For debugging purposes.
    """
    print(f"{name}:")
    print("-"*len(name))
    print(matrix)
    print()

def printNaNInf(array, name):
    """
    Prints the number and locations in given array where there
    are NaNs and infs.
    """
    nan_list = np.isnan(array).nonzero()
    inf_list = np.isinf(array).nonzero()
    print(f"{name} has {nan_list[0].shape[0]} NaNs at {nan_list}")
    print(f"{name} has {inf_list[0].shape[0]} Infs at {inf_list}")

def kickFunction(theta1, theta2, theta3):
    """
    Returns the kick part of floquet operator.
    """
    quasikick = (1 + ALPHA * np.cos(theta2) * np.cos(theta3))
    return np.exp(-1j * K * np.cos(theta1) * quasikick / HBAR)

def getInitialState():
    """
    Returns the initial state as tensor product of the 3 momentum spaces.
    """
    state1 = np.zeros(DIM, dtype=DTYPE)
    state1[N] = 1 # State |0>

    state2 = np.ones(DIM, dtype=DTYPE) / np.sqrt(DIM)
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
                            fourier = fourier_view[m3-n3+shift, m1-n1+shift, m2-n2+shift]
                            row = (m1 + N) * DIMSQ + (m2 + N) * DIM + (m3 + N)
                            col = (n1 + N) * DIMSQ + (n2 + N) * DIM + (n3 + N)
                            F_view[row, col] = phase * fourier

    return F


def getFloquetOperator():
    """
    Returns the floquet operator for the 3d quasiperiodic kicked rotor.
    """
    fourier_coeffs = getKickFourierCoeffs(kickFunction)
    # fourier_coeffs[np.abs(fourier_coeffs) < EPSILON] = 0
    # printNaNInf(fourier_coeffs, "fourier_coeffs")

    F = getDenseFloquetOperator(fourier_coeffs)
    # printNaNInf(F, "F")
    F.real[np.abs(F.real) < EPSILON] = 0
    F.imag[np.abs(F.imag) < EPSILON] = 0
    # np.nan_to_num(F, copy=False, nan=0)
    # printNaNInf(F, "F")
    # printMatrix(F, "F")
    # sign, slogdet = np.linalg.slogdet(F)
    # print(f"det(F) = {sign} x exp({slogdet})")
    # print(f"Eigenvalues of F are {np.sort(np.abs(eigvals(F)))}")
    Fh = np.conjugate(F.T)
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

def getMomentumDistribution(rho1):
    """
    Returns the probability distribution of the momentum p1.
    """
    return np.diag(rho1)

def getAvgMomentumSq(rho1):
    """
    Returns the ensemble average of the momentum p1^2.
    """
    return np.sum(np.diag(rho1) * np.arange(-N, N+1)**2)

def plotEntropies(ax, entropies):
    """
    Plots the entanglement entropies as a function of time.
    """
    t = np.arange(1, TIMESTEPS+1)

    ax.semilogy(t, entropies, marker="o")
    ax.set_xlabel("Time (t)")
    ax.set_ylabel(r"Entanglement Entropy ($\sigma$)")

def plotMomentumDistribution(ax, momentum_dist):
    """
    Plots the distribution of the momentum p1 at the end
    of the simulation.
    """
    p = np.arange(-N, N+1)
    ax.semilogy(p, momentum_dist, marker="o")
    ax.set_xlabel(r"$m_1$")
    ax.set_ylabel(r"$P(p_1 = m_1 \hbar)$")

def plotMomentumSqAvg(ax, momentum_sq_avgs):
    """
    Plots the average momentum p1^2 with time.
    """
    t = np.arange(1, TIMESTEPS+1)
    ax.plot(t, momentum_sq_avgs, marker="o")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$[m_1^2]$")

def plotter(entropies, momentum_sq_avgs, momentum_dist):
    """
    Plots all the computed quantities, namely entanglement
    entropy, average momentum p1 and distribution of the
    momentum p1 at the end.
    """
    fig, ax = plt.subplots(nrows=2, ncols=2)

    plotEntropies(ax[0, 0], entropies)
    plotMomentumDistribution(ax[0, 1], momentum_dist)
    plotMomentumSqAvg(ax[1, 0], momentum_sq_avgs)

    plt.tight_layout()
    plt.savefig("plots/entanglement_entropy.png")


def main():
    """
    The main function which handles the entire execution sequence.
    """
    np.seterr(all="raise")
    seterr(all="raise") # scipy.special.seterr

    print("Starting to compute F...")
    F, Fh = getFloquetOperator() # F is sparse CSR, Fh is sparse CSC

    print("Floquet operator computation over. Density operator starting...")
    rho = getInitialDensity()
    rho1 = partialTrace(rho)

    print("Trace of rho:", np.trace(rho))
    print("Trace of rho1:", np.trace(rho1))
    print("von Neumann Entropy:", getEntanglementEntropy(rho1))
    print()

    momentum_sq_avgs = np.empty(TIMESTEPS)
    entropies = np.empty(TIMESTEPS)
    print("Starting evolution...")

    for t in range(TIMESTEPS):
        print(f"Iteration {t} starting...")

        rho1 = partialTrace(rho)
        # np.nan_to_num(rho1, copy=False, nan=0)

        # printNaNInf(rho1, "rho1")

        print(f"Trace of rho1: {np.trace(rho1)}")


        printNaNInf(rho, "rho")

        entropy = getEntanglementEntropy(rho1)
        print(f"Calculated von Neumann entropy is: {entropy}")
        entropies[t] = entropy.real
        momentum_sq_avgs[t] = getAvgMomentumSq(rho1).real

        rho = evolve(rho, F, Fh)
        rho /= np.trace(rho)
        # np.nan_to_num(rho, copy=False, nan=0)

        print()

    rho1 = partialTrace(rho)
    momentum_dist = getMomentumDistribution(rho1)
    plotter(entropies, momentum_sq_avgs, momentum_dist)

if __name__ == "__main__":
    main()
    plt.show()
