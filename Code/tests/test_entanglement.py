import numpy as np
import kickedrotor.bipartite_entanglement as rotor

def test_momentum_to_angle():
    """
    a[m1, m2, m3] -> b[j1, j2, j3]
    b[j1, j2, j3] = sum_[m1, m2, m3] a[m1, m2, m3] * exp(i m . j 2pi/DIMF)
    """
    a = np.random.normal(loc=0, scale=1, size=(rotor.DIM, rotor.DIM, rotor.DIM))
    j = np.arange(rotor.DIMF)
    j1, j2, j3 = np.meshgrid(j, j, j, indexing="ij")
    b = np.empty((rotor.DIMF, rotor.DIMF, rotor.DIMF), dtype=rotor.DTYPE)

    for m1 in range(-rotor.N, rotor.N + 1):
        for m2 in range(-rotor.N, rotor.N + 1):
            for m3 in range(-rotor.N, rotor.N + 1):
                b += a[m1+rotor.N, m2+rotor.N, m3+rotor.N] * \
                    np.exp(1j * (m1*j1 + m2*j2 + m3*j3) * (2*np.pi/rotor.DIMF))
    b /= rotor.DIM**(3/2)
    observed = rotor.momentumToAngle(a.reshape(rotor.DIM**3))
    np.testing.assert_allclose(observed, b.reshape(rotor.DIMF**3))


def test_identity():
    state = np.random.normal(size=(rotor.DIM**3))
    state /= np.linalg.norm(state)
    observed = rotor.angleToMomentum(rotor.momentumToAngle(state))
    assert np.allclose(state, observed)

def test_partialTrace():
    rho1 = np.diag(np.arange(rotor.DIM) / np.sum(np.arange(rotor.DIM)))
    rho2 = np.ones((rotor.DIM, rotor.DIM)) / rotor.DIM
    rho3 = np.zeros((rotor.DIM, rotor.DIM))
    rho3[0, 0] = 0.5
    rho3[1, 1] = 0.5
    rho = np.kron(np.kron(rho1, rho2), rho3)
    # print("rho.shape before reshape:", rho.shape)
    rho = rho.reshape(rotor.DIM**3, rotor.DIM**3)
    obtained_rho1 = rotor.partialTrace(rho)
    np.testing.assert_allclose(obtained_rho1, rho1)
    np.testing.assert_allclose(np.trace(obtained_rho1),1)

def test_entropy():
    # Testing for pure state
    x = np.zeros((10, 10))
    x[0, 0] = 1
    assert rotor.vonNeumannEntropy(x) == 0

    # Testing for completely mixed state
    x = np.eye(10) / 10
    entropy = rotor.vonNeumannEntropy(x)
    np.testing.assert_allclose(-np.log(0.1), entropy)
