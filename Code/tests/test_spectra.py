"""
Author: Aditya Chincholi
Purpose: To debug and unittest the spectral statistics functions
"""
import numpy as np
import scipy.stats as stats
import kickedrotor.spectral_statistics as spectra

def test_phases():
    phases = np.linspace(-np.pi*0.85, np.pi*0.5, 100)
    phases.sort()
    M = np.diag(np.exp(1j * phases))
    U = stats.unitary_group.rvs(100)
    Mp = U.dot(M.dot(U.conjugate().T))
    observed_phases, eigs, num_dicarded \
        = spectra.getPhaseSpectrum(Mp, tol=1e-3, discard=True)
    np.testing.assert_allclose(observed_phases, phases)

def test_spacings():
    x = np.arange(100)
    y = np.arange(0, 200, 2)
    spacings, ratios = spectra.getSpacingRatios(x + 1j*y)
    np.testing.assert_allclose(spacings, 1)

    x = np.cumsum(np.arange(0, 100))
    expected_spacings = np.arange(1, 100) / np.mean(np.arange(1, 100))
    expected_ratios = np.arange(2, 100) / np.arange(1, 99)
    spacings, ratios = spectra.getSpacingRatios(x)
    np.testing.assert_allclose(spacings, expected_spacings)
    np.testing.assert_allclose(ratios, expected_ratios)
