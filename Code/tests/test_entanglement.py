import numpy as np
import scipy.fft as fft
from scipy.sparse import csr_matrix
from kickedrotor import bipartite_entanglement_pure_python as rotor

HBAR = 1
K = 5
ALPHA = 0.1
OMEGA2 = 0.9
OMEGA3 = 1.2
N = 2
DIM = 2*N + 1

def test_initialstate():
    main_vunit = np.ones((DIM, DIM)) / DIM
    zero_vunit = np.zeros((DIM, DIM))
    units = (zero_vunit,)*N + (main_vunit,) + (zero_vunit,)*N
    state = np.vstack(units).reshape(DIM, DIM, DIM)
    obtained_state = rotor.getInitialState()
    np.testing.assert_almost_equal(state, obtained_state)
    # print(result)

def test_fourier():
    y = rotor.getKickFourierCoeffs(rotor.kickFunction)
    y_raw = fft.ifftshift(y)
    x = fft.ifftn(y_raw)
    theta = np.arange(2*DIM) * 2 * np.pi / (2*DIM)
    theta1, theta2, theta3 = np.meshgrid(theta, theta, theta)
    expected = rotor.kickFunction(theta1, theta2, theta3)
    np.testing.assert_almost_equal(x, expected)

def test_entropy():
    # Testing for pure state
    x = np.zeros((10, 10))
    x[0, 0] = 1
    assert rotor.getEntanglementEntropy(x) == 0

    # Testing for completely mixed state
    x = np.eye(10) / 10
    entropy = rotor.getEntanglementEntropy(x)
    np.testing.assert_almost_equal(-np.log(0.1), entropy)

def test_partialTrace():
    rho1 = np.diag(np.arange(DIM) / np.sum(np.arange(DIM)))
    rho2 = np.ones((DIM, DIM)) / DIM
    rho3 = np.zeros((DIM, DIM))
    rho3[0, 0] = 1

    rho = np.tensordot(np.tensordot(rho1, rho2, axes=0), rho3, axes=0)
    # print("rho.shape before reshape:", rho.shape)
    rho = rho.reshape(DIM**3, DIM**3)
    obtained_rho1 = rotor.partialTrace(rho)
    np.testing.assert_almost_equal(obtained_rho1, rho1)
    np.testing.assert_almost_equal(np.trace(obtained_rho1),1)

def prepareProjector(phi2, phi3):
    m = np.arange(-N, N+1)
    state2 = np.exp(-1j * phi2 * m) / np.sqrt(N)
    state3 = np.exp(-1j * phi3 * m) / np.sqrt(N)
    state = np.tensordot(state2, state3, axes=0)
    projector = np.tensordot(np.eye(DIM), np.outer(state, state), axes=0)
    return projector

def denseFloquetOperator(phi2, phi3):
    # Kick strength is k (1 + α cos(ω_2) cos(ω3))
    kick_strength = K * (1 + ALPHA * np.cos(OMEGA2 + phi2) \
                    * np.cos(OMEGA3 + phi3))
    n = np.arange(-N, N+1)
    colgrid, rowgrid = np.meshgrid(n, n)
    F = np.exp(-1j * HBAR * colgrid**2 / 2) \
        * jv(colgrid - rowgrid, -kick_strength / HBAR) \
        * (1j)**(colgrid - rowgrid)
    return F

def get1dFloquetOperator(t, base_strength, **params):
    F = denseFloquetOperator(t, base_strength, **params)
    F[np.abs(F) < EPSILON] = 0
    return np.matrix(F, dtype=np.complex64)

def test_floquetOperator():
    F, Fh = rotor.getFloquetOperator()
    projector_right = prepareProjector(0, 0)
    projector_left = prepareProjector(OMEGA2, OMEGA3)
    obtained_F1d = csr_matrix.dot(projector_left, F.dot(projector_right))
    expected_F1d = get1dFloquetOperator(0, 0)
    np.testing.assert_almost_equal(obtained_F1d, expected_F1d)
