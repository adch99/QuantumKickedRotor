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
from scipy.special import xlogy, seterr
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse import eye as sparse_eye
from scipy.linalg import eigvals
import matplotlib.pyplot as plt
import kickedrotor.quasiperiodic_rotor_3d as floquet
from kickedrotor.params import *

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

def getFloquetOperator():
    """
    Returns the floquet operator for the 3d quasiperiodic kicked rotor.
    """
    F = floquet.getFloquetOperator()
    F.real[np.abs(F.real) < EPSILON] = 0
    F.imag[np.abs(F.imag) < EPSILON] = 0
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
