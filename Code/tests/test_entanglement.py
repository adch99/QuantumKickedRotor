import numpy as np
import scipy.fft as fft
from kickedrotor import bipartite_entanglement as rotor

DIM = rotor.DIM
N = rotor.N

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

def test_partialtrace():
    rho1 = np.diag(np.arange(DIM) / np.sum(np.arange(DIM)))
    rho2 = np.ones((DIM, DIM)) / DIM
    rho3 = np.zeros((DIM, DIM))
    rho3[0, 0] = 1

    rho = np.tensordot(np.tensordot(rho1, rho2, axes=0), rho3, axes=0)
    # print("rho.shape before reshape:", rho.shape)
    rho = rho.reshape(DIM**3, DIM**3)
    obtained_rho1 = rotor.partialTrace(rho)
    np.testing.assert_almost_equal(obtained_rho1, rho1)
