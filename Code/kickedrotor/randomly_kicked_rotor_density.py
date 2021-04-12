import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit
from sys import argv
sns.set()

# Physical Constants
# HBAR = 1.0545718e-34
HBAR = 1
TAU = 1 * HBAR
K = 5 / HBAR

# Program Constants
N = 1000
DIM = 2*N + 1 # [-N, N]
EPSILON = 1e-6
SAMPLES = 1
TIMESPAN = 500

def printMatrix(matrix, name):
    print(f"{name}:")
    print("-"*len(name))
    print(matrix)
    print()

def getFplus():
    multiplier = np.exp(-1j*K) * np.exp(-1j * TAU * HBAR * np.arange(-N, N+1)**2 / 2)
    return np.matrix(np.diag(multiplier), dtype=np.complex64)

def getFminus():
    multiplier = np.exp(1j*K) * np.exp(-1j * TAU * HBAR * np.arange(-N, N+1)**2 / 2)
    return np.matrix(np.diag(multiplier), dtype=np.complex64)

def uniformDistOnNSphere():
    """
    Generate SAMPLES number of random gaussian
    vectors in [-1, 1]^DIM, then normalize it.
    Mathematically, it works out to be
    uniform on the sphere.
    """
    bases = np.random.normal(loc=0, scale=1, size=(DIM, SAMPLES))
    norms = np.linalg.norm(bases, axis=0)
    return bases / norms

def zeroMomentumState():
    state = np.zeros(DIM)
    state[N + 0] = 1
    return state

def densityOperator():
    # states = uniformDistOnNSphere()
    # states = np.eye(DIM, SAMPLES) / SAMPLES
    states = np.array([zeroMomentumState()]).T
    # print("states.shape:", states.shape)
    rho = np.zeros((DIM, DIM))
    for i in range(SAMPLES):
         state = np.matrix(states[:, i]).T
         rho += state.dot(state.getH()) * (1/SAMPLES)
    return np.matrix(rho, dtype=np.complex64)


def L2Operator():
    L2 = HBAR**2 * np.diag(np.arange(-N, N+1)**2)
    return np.matrix(L2, dtype=np.complex64)

@jit
def EnsembleAvg(rho, T):
    """
    Ensemble average of the T operator.
    Given by tr(T*rho).
    """
    product = np.dot(rho, T)
    return np.trace(product)

def Ldistribution(rho):
    p = np.zeros(DIM)
    for n in range(-N, N+1):
        basis_vector = np.zeros(DIM)
        basis_vector[n+N] = 1
        basis_vector = np.matrix(basis_vector).T
        p[n+N] = (basis_vector.T.dot(np.dot(rho, basis_vector))).item().real
    return p

@jit
def evolve(rho, F, Fh):
    """
    Evolves the density operator by one
    time period step.
    """
    return F.dot(np.dot(rho, Fh))

def run():
    # Both the operators are sparse, rho is not.
    rho = densityOperator()
    L2 = L2Operator()
    L4 = L2.dot(L2) # L^4 operator
    Fplus = getFplus()
    Fplush = Fplus.getH()
    Fminus = getFminus()
    Fminush = Fminus.getH()
    L2ensembleavgs = np.zeros(TIMESPAN, dtype=np.complex64)
    L2ensemblevars = np.zeros(TIMESPAN, dtype=np.complex64)

    kick_sequence = np.random.choice([-1, 1], size=TIMESPAN, replace=True)

    for t, kick in enumerate(kick_sequence):
        print(f"Starting cycle no {t}...")
        L2avg = EnsembleAvg(rho, L2)
        L4avg = EnsembleAvg(rho, L4)
        L2var = L4avg - L2avg**2

        L2ensembleavgs[t] = L2avg
        L2ensemblevars[t] = L2var

        if kick == 1:
            rho = evolve(rho, Fplus, Fplush)
        else:
            rho = evolve(rho, Fminus, Fminush)

    probL = Ldistribution(rho)

    return L2ensembleavgs, L2ensemblevars, probL

def plot(avgs, vars, p):
    t = np.arange(TIMESPAN)
    n = np.arange(-N, N+1)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2,
        figsize=(16, 12))
    ax1.plot(t, avgs.real)
    ax1.set_yscale("log")
    ax1.set_ylabel(r"$[L^2]$")
    ax1.set_xlabel("t")

    ax3.plot(t, vars.real)
    # ax3.set_yscale("log")
    ax3.set_ylabel(r"$[\Delta(L^2)]$")
    ax3.set_xlabel("t")

    ax2.plot(n, np.log10(p))
    # ax2.scatter(n, p)
    # ax2.set_yscale("log")
    ax2.set_ylabel(r"$log_{10}(p(L = \hbar n))$")
    ax2.set_xlabel("n")
    title = f"Randomly Kicked Quantum Rotor [K={K}, TAU={TAU}, DIM={DIM}]"
    fig.suptitle(title)
    plt.tight_layout()
    filename = f"plots/randomly_kicked_plots/density.png"
    plt.savefig(filename)

if __name__ == "__main__":
    avgs, vars, p = run()
    plot(avgs, vars, p)
    # print("p:", p)
    plt.show()
