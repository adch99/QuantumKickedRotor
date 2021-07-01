import numpy as np
import kickedrotor.bipartite_entanglement_alt as rotor

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
