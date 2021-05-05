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

import sys
import numpy as np
import scipy.fft as fft
from scipy.special import xlogy
from scipy.sparse import csr_matrix, csc_matrix
# from numba import jit
import matplotlib.pyplot as plt
import cython
# Scientific Constants
HBAR = 2.89
K = 5
ALPHA = 0.5
OMEGA2 = 2 * np.pi * np.sqrt(5)
OMEGA3 = 2 * np.pi * np.sqrt(13)

# Program Constants
N = 1
DIM = 2 * N + 1
EPSILON = 1e-6
TIMESTEPS = 5
FSAMPLES = max(64, 2*DIM)

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
    quasikick = (1 + ALPHA * np.cos(theta2) * np.cos(theta3))
    return np.exp(-1j * K * np.cos(theta1) * quasikick / HBAR)

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
    theta = np.arange(FSAMPLES) * 2 * np.pi / FSAMPLES
    theta1, theta2, theta3 = np.meshgrid(theta, theta, theta)
    x = func(theta1, theta2, theta3)
    y = fft.fftshift(fft.fftn(x, norm="ortho")) * (2 * np.pi / FSAMPLES)**1.5
    return y.astype(np.complex64)


# @jit(nopython=True, parallel=True)
def getDenseFloquetOperator(fourier_coeffs):
    """
    Returns the dense version of the floquet operator.
    """
    shift = int(FSAMPLES / 2 + 1)
    norm = 1 / (2 * np.pi)**1.5
    m = np.arange(-N, N+1, dtype=int)
    m1, m2, m3, n1, n2, n3 = np.meshgrid(*(m,)*6)
    F = np.exp(-1j * (HBAR * n1**2 / 2) + n2 * OMEGA2 + n3 * OMEGA3) \
        * fourier_coeffs[m1-n1+shift, m2-n2+shift, m3-n3+shift] * norm
    return F

def getFloquetOperator():
    """
    Returns the floquet operator for the 3d quasiperiodic kicked rotor.
    """
    fourier_coeffs = getKickFourierCoeffs(kickFunction)
    plt.semilogy(np.arange(-FSAMPLES/2, FSAMPLES/2), np.abs(fourier_coeffs[2*N, 2*N, :]))
    F = getDenseFloquetOperator(fourier_coeffs).reshape((DIM**3, DIM**3))
    print("det(F) =", np.abs(np.linalg.det(F)))
    F[np.abs(F) < EPSILON] = 0
    with np.printoptions(threshold=np.inf, suppress=True, precision=4):
        printMatrix(F, "F")
    Fh = np.conjugate(F.T)
    return csr_matrix(F), csr_matrix(Fh)

def evolve(rho, F, Fh):
    """
    Evolves the density matrix by one unit time step.
    """
    return csc_matrix.dot(F.dot(rho), Fh)

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
    ax.plot(t, entropies, marker="o")
    ax.set_xlabel("Time (t)")
    ax.set_ylabel(r"Entanglement Entropy ($\sigma$)")
    plt.savefig("plots/entanglement_entropy.png")

def main():
    """
    The main function which handles the entire execution sequence.
    """
    F, Fh = getFloquetOperator() # F is sparse CSR, Fh is sparse CSC
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
