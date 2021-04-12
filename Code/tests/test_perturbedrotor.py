from nose2.tools import params
import numpy as np
from kickedrotor import perturbed_quantum_kicked_rotor as rotor
from scipy.special import jv

def setup():
    print("Setting up the test...")

def teardown():
    print("Tearing down the test...")

@params((0,0), (0,0.1), (0.1,0))
def test_FloquetOperator(dk=0, dtau=0):
    print(f"Running test for deltak = {dk}, deltatau = {dtau}...    ")
    tau = rotor.TAU + dtau
    k = rotor.K + dk
    N = rotor.N
    n = np.arange(-N, N+1)
    colgrid, rowgrid = np.meshgrid(n, n)
    F = np.exp(-1j * tau * colgrid**2 / 2) * jv(colgrid - rowgrid, -k) * (1j)**(colgrid - rowgrid)
    F[np.abs(F) < rotor.EPSILON] = 0

    bound = rotor.findBesselBounds()
    sampleF = rotor.floquetOperator(bound, dk, dtau)
    np.testing.assert_allclose(sampleF, np.matrix(F))
