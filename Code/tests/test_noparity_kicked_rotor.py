import numpy as np
from scipy.special import jv
import kickedrotor.quantum_kicked_rotor_noparity as noparity
import kickedrotor.perturbed_quantum_kicked_rotor as parity

def test_total():
    expected = parity.denseFloquetOperator()
    observed = noparity.floquetMatrix(beta = 0, hbar = 1, tau = 1, k = 5)
    # print(observed / expected)
    np.testing.assert_allclose(observed, expected, atol=1e-7)

def test_momentum():
    p = np.arange(-noparity.N, noparity.N+1)
    mm, nn = np.meshgrid(p, p, indexing="ij")
    expected = np.exp(-1j * nn**2 / 2)
    observed = noparity.momentumPhase(tau = 1, hbar = 1)
    np.testing.assert_allclose(observed, expected, atol=1e-7)

def test_angular():
    n = np.arange(-noparity.N, noparity.N+1)
    mm, nn = np.meshgrid(n, n, indexing="ij")
    expected = jv(nn - mm, -5) * (1j)**(nn - mm)
    observed = noparity.angularPhase(beta = 0, hbar = 1, k = 5)
    np.testing.assert_allclose(observed, expected, atol=1e-7)

def test_fourier():
    p = np.arange(-2*noparity.N, 2*noparity.N+1)
    observed = noparity.fourierCoeffs(noparity.f, beta = 0, hbar = 1, k = 5)
    expected = jv(p, -5) * (1j)**p
    np.testing.assert_allclose(observed, expected, atol=1e-7)

def test_unitarity():
    F = noparity.floquetMatrix()
    F_h = (F.T).conjugate()
    observed = np.dot(F_h, F)
    expected = np.eye(noparity.DIM)
    diff = np.abs(observed - expected)
    print(f"Max diff: {diff.max()}\tMean diagonal diff: {np.mean(np.diag(diff))}")
    np.testing.assert_allclose(observed, expected, atol=1e-7)
