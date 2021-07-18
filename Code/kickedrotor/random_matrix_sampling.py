import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats
import scipy.linalg as linalg

# Quaternions
E0 = np.eye(2)
Ex = np.array(
    [[0, 1j],
    [1j, 0]])
Ey = np.array(
    [[0, 1],
    [-1, 0]])
Ez = np.array(
    [[1j, 0],
    [0, -1j]])
E = [E0, Ex, Ey, Ez]


def randomGOE(n):
    # return stats.ortho_group.rvs(n) # This also works
    A = np.random.normal(loc=0, scale=1, size=(n,n))
    return (A + A.T) / 2

def randomGUE(n):
    return stats.unitary_group.rvs(n)

def randomGSE(n):
    if n % 2 == 1:
        raise ValueError("n must be even for a CSE matrix.")
        return None

    half = int(n / 2)
    V = np.zeros((n, n), dtype=np.complex128)
    for i, Ei in enumerate(E):
        Hi = np.random.normal(loc=0, scale=1, size=(half, half))
        V += np.kron(Hi, Ei)
    U = V + V.T.conj()
    return U / np.sqrt(8*n)


def randomCOE(n):
    A = randomGOE(n)
    O = (A + A.T.conj()) / 2
    T, Z = linalg.schur(O, output="complex")
    S = np.diag((-1)**np.random.randint(2, size=(n,)))
    return S.dot(Z)

def randomCUE(n):
    """
    A Random CUE matrix distributed with Haar measure.

    Taken from Mezzadri, F. How to generate random matrices
    from the classical compact groups. arXiv:math-ph/0609050 (2007).

    """
    x = np.random.normal(loc=0, scale=1, size=(n,n))
    y = np.random.normal(loc=0, scale=1, size=(n,n))
    z = (x + 1j*y) / np.sqrt(2.0)
    q,r = linalg.qr(z)
    d = np.diag(r)
    ph = np.diag(d / np.abs(d))
    M = np.dot(q, ph)
    return M

def randomCOE(n):
    W = randomCUE(n)
    return np.dot(W, W.T)

def randomCSE(n):
    if n % 2 == 1:
        raise ValueError("n must be even for a CSE matrix.")
        return None

    half = int(n / 2)
    J = np.zeros((n,n))
    J[half:, 0:half] = -np.eye(half)
    J[0:half, half:] = np.eye(half)
    W = randomCUE(n)
    U = -np.linalg.multi_dot([W, J, W.T, J])
    return U


def checkUnitarity(F):
    F_h = (F.T).conjugate()
    obs = F.dot(F_h)
    diff = np.abs(obs - np.eye(F.shape[0]))
    obs_abs = np.abs(obs)
    print("Mean obs:", np.mean(np.diag(obs_abs)))
    print("Mean diff:", np.mean(diff))
    print("Max diff:", diff.max())
    norm = mpl.colors.SymLogNorm(linthresh=1e-9, vmin=obs_abs.min(),
        vmax=obs_abs.max(), base=10)
    plt.matshow(obs_abs, norm=norm, cmap="Purples_r")
    plt.colorbar()
