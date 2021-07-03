"""
Author: Aditya Chincholi
Purpose: To plot the spectrum of the floquet operator
        for the 3d quasiperiodic kicked rotor.
"""

import numpy as np
from scipy.linalg import eigvals as denseEigvals
from scipy.sparse.linalg import eigs as sparseEigs
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import kickedrotor.bipartite_entanglement as rotor

EPSILON = 1e-6
N = 10
DIM = 2*N + 1

def sparsifyFloquetOperator(F):
    """
    Returns the floquet operator for the 3d quasiperiodic kicked rotor.
    This version of the function is for testing only and it returns
    the both the sparse and dense versions of the floquet operator.
    """
    F.real[np.abs(F.real) < EPSILON] = 0
    F.imag[np.abs(F.imag) < EPSILON] = 0
    return csr_matrix(F)

def getEigenvalues():
    """
    Retrieves the floquet operator and returns the eigenvalues
    for the dense and sparse forms of the operator.
    """
    fourier_coeffs = rotor.getKickFourierCoeffs(rotor.kickFunction)
    F = rotor.getDenseFloquetOperator(fourier_coeffs)
    del fourier_coeffs
    # F_sparse = sparsifyFloquetOperator(F)

    eigvals_dense = denseEigvals(F)
    print(f"No of dense eigvals: {eigvals_dense.shape}")
    # eigvals_sparse = sparseEigs(F_sparse, return_eigenvectors=False)
    eigvals_sparse = None

    return eigvals_dense, eigvals_sparse

def plotValues(eigvals_dense, eigvals_sparse):
    """
    Plots the absolute value of the spectrum of the dense and
    sparse forms of the floquet operator.
    """
    abs_dense = np.sort(np.abs(eigvals_dense))
    # abs_sparse = np.sort(np.abs(eigvals_sparse))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    ax.hist(abs_dense, label="Dense Form", density=True)
    # ax.hist(abs_sparse, label="Sparse Form", density=True)
    # ax.legend()
    ax.set_yscale("log")
    ax.set_xlabel("Absolute value of Eigenvalue")
    ax.set_ylabel("Density of values")
    ax.set_title("Spectrum of Floquet Operator Forms")

    plt.savefig(f"plots/floquet_spectrum_{DIM}.png")

if __name__ == "__main__":
    sns.set()
    sns.set_context("talk")
    eigvals_dense, eigvals_sparse = getEigenvalues()
    plotValues(eigvals_dense, eigvals_sparse)
    plt.show()
