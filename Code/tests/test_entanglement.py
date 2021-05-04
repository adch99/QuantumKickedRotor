import numpy as np
import scipy.fft as fft
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs as speigs
from scipy.special import jv
import matplotlib.pyplot as plt
from kickedrotor import bipartite_entanglement as rotor

HBAR = 2.89
K = 5
ALPHA = 0.5
OMEGA2 = 2 * np.pi * np.sqrt(5)
OMEGA3 = 2 * np.pi * np.sqrt(13)
N = 1
DIM = 2*N + 1
EPSILON = 1e-6

def test_initialstate():
    main_vunit = np.ones(DIM**2) / DIM
    zero_vunit = np.zeros(DIM**2)
    units = (zero_vunit,)*N + (main_vunit,) + (zero_vunit,)*N
    state = np.hstack(units)
    obtained_state = rotor.getInitialState()
    np.testing.assert_allclose(state, obtained_state)
    # print(result)

def notest_fourier():
    y = rotor.getKickFourierCoeffs(rotor.kickFunction)
    y_raw = fft.ifftshift(y)
    x = fft.ifftn(y_raw, norm="ortho")
    theta = np.arange(2*DIM) * 2 * np.pi / (2*DIM)
    theta1, theta2, theta3 = np.meshgrid(theta, theta, theta)
    expected = rotor.kickFunction(theta1, theta2, theta3)
    np.testing.assert_allclose(x, expected)

def test_entropy():
    # Testing for pure state
    x = np.zeros((10, 10))
    x[0, 0] = 1
    assert rotor.getEntanglementEntropy(x) == 0

    # Testing for completely mixed state
    x = np.eye(10) / 10
    entropy = rotor.getEntanglementEntropy(x)
    np.testing.assert_allclose(-np.log(0.1), entropy)

def test_partialTrace():
    rho1 = np.diag(np.arange(DIM) / np.sum(np.arange(DIM)))
    rho2 = np.ones((DIM, DIM)) / DIM
    rho3 = np.zeros((DIM, DIM))
    rho3[0, 0] = 0.5
    rho3[1, 1] = 0.5
    rho = np.kron(np.kron(rho1, rho2), rho3)
    # print("rho.shape before reshape:", rho.shape)
    rho = rho.reshape(DIM**3, DIM**3)
    obtained_rho1 = rotor.partialTrace(rho)
    np.testing.assert_allclose(obtained_rho1, rho1)
    np.testing.assert_allclose(np.trace(obtained_rho1),1)


def kickFunction1dRotor(theta1, theta2, theta3):
    return np.exp(-1j * K * np.cos(theta1) / HBAR)

def notest_fourierBessel():
    fourier_coeffs = rotor.getKickFourierCoeffs(kickFunction1dRotor)
    fourier_sum = np.sum(fourier_coeffs, axis=(1,2)) / (2*DIM - 1)
    p = np.arange(-2*N, 2*N)
    expected = 2 * np.pi * (1j)**(-p) * jv(-p, -K / HBAR)

    np.testing.assert_allclose(fourier_sum, expected)

# def test_complexExp():
#     x = np.linspace(0, 2*np.pi, 100)
#     y = rotor.complex_exp(x)
#     expected = np.cos(x) + 1j*np.sin(x)
#     np.testing.assert_allclose(y, expected)

# def test_floquetOperator():
#     F, Fh = rotor.getFloquetOperator()
#     k = 5 # k < DIM**3 - 1
#     eigvals, eigvecs = speigs(F, k=k)
#     np.testing.assert_allclose(np.abs(eigvals), np.ones(k))
