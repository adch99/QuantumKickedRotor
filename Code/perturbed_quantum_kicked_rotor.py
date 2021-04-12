import numpy as np
from scipy.special import jv # Bessel function of first kind
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs as speigs
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit
from datetime import date
from sys import argv
sns.set()

# Physical Constants
# HBAR = 1.0545718e-34
HBAR = 1
TAU = 1 * HBAR
K = 5 / HBAR
DELTAK = 0 # 1e-2 / HBAR
DELTATAU = 1e-2 / HBAR

# Program Constants
N = 1000
DIM = 2*N + 1 # [-N, N]
EPSILON = 1e-6
SAMPLES = 1
TIMESPAN = 500

def cmdArgSetter(argv):
    argc = len(argv)

    if argc == 1: return
    for index, arg in enumerate(argv):
        if arg == "-n" and argc > index+1:
            global N
            global DIM
            N = int(argv[index+1])
            DIM = 2*N + 1
        if arg == "-t" and argc > index+1:
            global TIMESPAN
            TIMESPAN = int(argv[index+1])
        if arg == "--deltak" and argc > index+1:
            global DELTAK
            DELTAK = float(argv[index+1])
        if arg == "--deltatau" and argc > index+1:
            global DELTATAU
            DELTATAU = float(argv[index+1])

    for var, name in zip([N, TIMESPAN, DELTAK, DELTATAU], ["N", "TIMESPAN", "DELTAK", "DELTATAU"]):
        print(f"{name}: {var}")


def printMatrix(matrix, name):
    print(f"{name}:")
    print("-"*len(name))
    print(matrix)
    print()

def cis(theta):
    return np.cos(theta) + 1j*np.sin(theta)

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
    return np.matrix(rho)


def denseFloquetOperator(deltak=0, deltatau=0):
    """
    Returns the Floquet operator in a dense form in order
    to circumvent numba's issues with sparse matrices.
    """
    tau = TAU + deltatau
    k = K + deltak
    n = np.arange(-N, N+1)
    colgrid, rowgrid = np.meshgrid(n, n) # rowgrid, colgrid
    F = np.exp(-1j * tau * colgrid**2 / 2) * jv(colgrid - rowgrid, -k) * (1j)**(colgrid - rowgrid)
    return F

def floquetOperator(deltak=0, deltatau=0):
    """
    Returns the floquet operator in sparse matrix form.
    """
    F = denseFloquetOperator(deltak, deltatau)
    F[np.abs(F) < EPSILON] = 0
    # sparsity = 1 - np.count_nonzero(F)/(2*N + 1)**2
    # print("Sparsity: %.10f" % sparsity)
    # print("Total F Sum: %.10f" % np.sum(F))
    F_sparse = csr_matrix(F)
    return F_sparse

def L2Operator():
    L2 = HBAR**2 * np.diag(np.arange(-N, N+1)**2)
    L2_sparse = csr_matrix(L2)
    return L2_sparse

def EnsembleAvg(rho, T):
    """
    Ensemble average of the T operator.
    Given by tr(T*rho).
    """
    product = csr_matrix.dot(rho, T)
    return ( product.diagonal(0) ).sum()

def Ldistribution(rho):
    p = np.zeros(DIM)
    for n in range(-N, N+1):
        basis_vector = np.zeros(DIM)
        basis_vector[n+N] = 1
        basis_vector = csr_matrix(basis_vector).T
        p[n+N] = (basis_vector.T.dot(csr_matrix.dot(rho, basis_vector))).item().real
    return p

def evolve(rho, F, Fh):
    """
    Evolves the density operator by one
    time period step.
    """
    return F.dot(csr_matrix.dot(rho, Fh))

def getPeriodPerturbations(cumulative=True):
    """
    Returns a sequence of perturbations to the period of kicking.
    cumulative = True means we don't correct for the previous
    perturbation. This leads to a slow drift.
    cumulative = False means we correct for the previous perturbation
    in the next one.
    """
    periodperturbations = np.random.uniform(-DELTATAU, DELTATAU, TIMESPAN)
    if not cumulative:
        periodperturbations[1:] -= periodperturbations[:-1]
    return periodperturbations


def run():
    # Both the operators are sparse, rho is not.
    rho = densityOperator()
    L2 = L2Operator()
    L4 = L2.dot(L2) # L^4 operator
    L2ensembleavgs = np.zeros(TIMESPAN, dtype=np.complex64)
    L2ensemblevars = np.zeros(TIMESPAN, dtype=np.complex64)
    if DELTAK != 0:
        kickperturbations = np.random.uniform(-DELTAK, DELTAK, TIMESPAN)
    else:
        kickperturbations = np.zeros(TIMESPAN)
    if DELTATAU != 0:
        periodperturbations = getPeriodPerturbations(cumulative=False)
    else:
        periodperturbations = np.zeros(TIMESPAN)

    standardrun = (DELTAK == 0 and DELTATAU == 0)
    if standardrun:
        F = floquetOperator()
        Fh = F.getH()

    for t in range(TIMESPAN):
        print(f"Starting cycle no {t}...")
        L2avg = EnsembleAvg(rho, L2)
        L4avg = EnsembleAvg(rho, L4)
        L2var = L4avg - L2avg**2

        L2ensembleavgs[t] = L2avg
        L2ensemblevars[t] = L2var

        if not standardrun:
            F = floquetOperator(deltak=kickperturbations[t],
                deltatau=periodperturbations[t])
            Fh = F.getH()
        rho = evolve(rho, F, Fh)

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
    title = f"Kicked Quantum Rotor [K={K}, TAU={TAU}, DIM={DIM}, T={TIMESPAN}, DELTAK={DELTAK}, DELTATAU={DELTATAU}]"
    fig.suptitle(title)
    plt.tight_layout()
    filename = f"plots/perturbedquantumkickedrotor_T{TIMESPAN}DIM{DIM}DK{DELTAK}DT{DELTATAU}_{date.today()}.png"
    plt.savefig(filename)

if __name__ == "__main__":
    cmdArgSetter(argv)
    avgs, vars, p = run()
    plot(avgs, vars, p)
    # print("p:", p)
    plt.show()
