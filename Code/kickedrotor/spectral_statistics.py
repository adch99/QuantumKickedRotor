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
import seaborn as sns

CSE_SPACING_CONST = 2**18 / (3**6 * np.pi**3)

def getPhaseSpectrum(F, tol=1e-9, discard=True):
    """
    Returns the phases of the eigenvalues of the given unitary matrix.
    """
    eigs = linalg.eigvals(F)
    amplitudes = np.abs(eigs)
    phases = np.angle(eigs)
    sort_key = np.argsort(phases)

    eigs = eigs[sort_key]
    amplitudes = amplitudes[sort_key]
    phases = phases[sort_key]

    error = np.abs(amplitudes - 1)
    retainable = (error < tol)
    num_discard = eigs.shape[0] - np.count_nonzero(retainable)
    if discard:
        return phases[retainable], eigs[retainable], num_discard
    else:
        return np.sort(phases), eigs

def getSpacingRatios(eigenphases):
    """
    Returns the spacings and spacing ratios of the eigenphases
    of the given matrix.
    """
    spacings = np.abs(np.diff(eigenphases))
    spacings *= 2 * np.pi / len(spacings)
    ratios = spacings[1:] / spacings[:-1]
    return spacings, ratios

def spacingPoisson(s):
    """
    Returns the expected Poisson distribution for level spacings
    in integrable systems.
    """
    return np.exp(-s)

def spacingSurmiseCUE(s):
    """
    Returns the Wigner surmise for the spacings distribution
    in CUE.
    """
    return (32/np.pi**2) * s**2 * np.exp(-s**2 * 4 / np.pi)

def spacingSurmiseCOE(s):
    """
    Returns the Wigner surmise for the spacing distribution
    in COE.
    """
    return (np.pi / 2) * s * np.exp(-s**2 * np.pi / 4)

def spacingSurmiseCSE(s):
    """
    Returns the Wigner surmise for the spacing distribution
    in CSE
    """
    return CSE_SPACING_CONST * s**4 * np.exp(-s**2 * 64 / (9 * np.pi))

def ratioPoisson(r):
    """
    Returns the distribution of level spacings ratio for integrable
    systems derived from the Poisson spacing distribution.
    """
    return 1 / (1 + r)**2

def ratioSurmiseCUE(r):
    """
    Returns the Wigner-like surmise for the spacing ratio
    distribution in CUE.
    """
    Z = 4 * np.pi / (81 * np.sqrt(3))
    return (1/Z) * (r + r**2)**2 / (1 + r + r**2)**4

def ratioSurmiseCOE(r):
    """
    Returns the Wigner-like surmise for the spacing ratio
    distribution in COE.
    """
    Z = 8 / 27
    return (1/Z) * (r + r**2)**1 / (1 + r + r**2)**2.5

def ratioSurmiseCSE(r):
    """
    Returns the Wigner-like surmise for the spacing ratio
    distribution in CSE.
    """
    Z = 4 * np.pi / (729 * np.sqrt(3))
    return (1/Z) * (r + r**2)**4 / (1 + r + r**2)**7


def plotRatios(ratios, ax):
    # lims = (ratios.min(), ratios.max())
    r = np.linspace(0, 5, 100)
    poisson = ratioPoisson(r)
    cue = ratioSurmiseCUE(r)
    coe = ratioSurmiseCOE(r)
    cse = ratioSurmiseCSE(r)
    ax.hist(ratios, density=True, bins=100,
        histtype="step", label="Calculated", range=(0, 5))
    ax.plot(r, poisson, label="Poisson")
    ax.plot(r, cue, label="CUE")
    ax.plot(r, coe, label="COE")
    ax.plot(r, cse, label="CSE")
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$P(r)$")
    # ax.set_yscale("log")
    ax.legend()

def plotSpacings(spacings, ax):
    # lims = (spacings.min(), spacings.max())
    s = np.linspace(0, 5, 100)
    N = len(spacings)
    poisson = N * spacingPoisson(s)
    cue = N * spacingSurmiseCUE(s)
    coe = N * spacingSurmiseCOE(s)
    cse = N * spacingSurmiseCSE(s)

    # counts, bins = np.histogram(spacings, cumulative=True)
    # total = sum(counts)
    # print("counts:", counts)
    # print("bins:", bins)
    # bin_width = bins[1] - bins[0]
    # counts = [count/total for count in counts]
    # ax.bar(x=bins, height=counts, width=bin_width, label="Calculated", filled=False)

    ax.hist(spacings, bins=100,
        range=(0,5), histtype="step", label="Calculated")
    ax.plot(s, poisson, label="Poisson")
    ax.plot(s, cue, label="CUE")
    ax.plot(s, coe, label="COE")
    ax.plot(s, cse, label="CSE")
    ax.set_xlabel(r"$s$")
    ax.set_ylabel(r"$W(s)$")
    # ax.set_yscale("log")
    ax.legend()
