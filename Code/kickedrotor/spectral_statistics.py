"""
Author: Aditya Chincholi
Purpose: Spectral Statistics
        A library containing functions to calculate and
        visualize the spectral statistics of a given
        floquet matrix. We focus mainly on the distribution of
        the ratio of consecutive spacings between eigenvalues.
"""

import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt

def getPhaseSpectrum(F, tol=1e-9, discard=True):
    """
    Returns the phases of the eigenvalues of the given unitary matrix.
    """
    eigs = linalg.eigvals(F)
    amplitudes = np.abs(eigs)
    phases = np.angle(eigs)
    error = np.abs(amplitudes - 1)
    retainable = (error < tol)
    num_discard = eigs.shape[0] - np.count_nonzero(retainable)
    if discard:
        return np.sort(phases[retainable]), num_discard
    else:
        return np.sort(phases)


def getSpacingRatios(eigenphases):
    """
    Returns the spacings and spacing ratios of the eigenphases
    of the given matrix.
    """
    spacings = np.diff(eigenphases)
    ratios = spacings[1:] / spacings[:-1]
    return spacings, ratios

def spacingSurmise(s):
    """
    Returns the value of P_W(s) the Wigner surmise for the GOE/COE
    level spacings distribution.
    """
    return (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4)

def ratioSurmise(r):
    """
    Returns the value of P_W(r), the Wigner-like surmise for the
    GOE/COE level spacing ratio distribution.
    """
    Z = 8 / 27
    return (1/Z) * (r + r**2)**1 / (1 + r + r**2)**2.5

def plotRatios(ratios, ax):
    lims = (0, 0.5)
    r = np.linspace(*lims, 100)
    surmise = ratioSurmise(r)
    ax.hist(ratios, density=True, bins=100, range=lims, histtype="step")
    ax.plot(r, surmise)
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$P(r)$")
    ax.set_yscale("log")

def plotSpacings(spacings, ax):
    lims = (spacings.min(), spacings.max())
    s = np.linspace(*lims, 100)
    surmise = spacingSurmise(s)
    ax.hist(spacings, density=True, bins=100, histtype="step")
    ax.plot(s, surmise)
    ax.set_xlabel(r"$s$")
    ax.set_ylabel(r"$P(s)$")
    ax.set_yscale("log")
