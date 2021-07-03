import numpy as np
import scipy.linalg as linalg
import seaborn as sns
import kickedrotor.perturbed_quantum_kicked_rotor as rotor

def getParityEigenvalues(desired_parity, F, P, atol=1e-3):
    """
    Given the desired parity, the floquet operator F, the parity
    operator P, return the eigenvalues of F corresponding to the
    desired parity.
    """
    if desired_parity == "even":
        even, odd = True, False
    elif desired_parity == "odd":
        even, odd = False, True
    else:
        print("desired_parity should be either \"even\" or \"odd\"")
        return None

    M = F + P
    eigvals, eigvecs = linalg.eig(M)

    # First we should check if indeed the eigenvectors are eigenvectors
    # of F and P.

    V = F.dot(eigvecs) / eigvecs

    # We check if the standard deviation is small wrt the mean
    F_eigvals = np.mean(V, axis=0)
    std_errors = np.std(np.abs(V), axis=0) / np.abs(F_eigvals)
    print(f"Max rel error: {std_errors.max()}\tMin rel error: {std_errors.min()}")

    U = P.dot(eigvecs) / eigvecs
    P_eigvals = np.mean(U, axis=0)
    # We check if the standard deviation is small wrt the mean
    std_errors = np.std(np.abs(U), axis=0) / np.abs(P_eigvals)
    print(f"Max rel error: {std_errors.max()}\tMin rel error: {std_errors.min()}")

    parity = np.empty(eigvals.shape, dtype=bool)
    for i in range(len(eigvals)):
        if np.isclose(P_eigvals[i], 1, atol=atol):
            parity[i] = True
        elif np.isclose(P_eigvals[i], -1, atol=atol):
            parity[i] = False
        else:
            print(f"No parity symmetry in #{i}")
            parity[i] = False

    return F_eigvals[parity ^ odd]

if __name__ == "__main__":
    F = rotor.denseFloquetOperator()
    P = np.fliplr(np.eye(2001))
    getParityEigenvalues("even", F, P)
