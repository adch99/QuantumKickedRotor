"""
Author: Aditya Chincholi
Purpose:
    To simulate a driven quantum kicked rotor and plot the results.
    H = L^2/2 + k cos(θ) Σ δ(t - nτ) + β cos(Ωt) where β << k. The
    intention is to have only a perturbative driving and investigate
    the effect of far from resonance and near resonance conditions
    between the kick frequency 2π/τ and the driving frequency Ω.
"""

from sys import argv
from datetime import date
import time
import numpy as np
from scipy.special import jv # Bessel function of first kind
# from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Physical Constants
# HBAR = 1.0545718e-34
HBAR = 1
TAU = 1 * HBAR
K = 5 / HBAR
DELTAK = 0 # 1e-2 / HBAR
DELTATAU = 0 #1e-2 / HBAR
BETA = 1e-2
OMEGA = 1

# Program Constants
N = 1000
DIM = 2*N + 1 # [-N, N]
EPSILON = 1e-9
SAMPLES = 1
TIMESPAN = 100

def cmdArgSetter(argv):
    """
    Sets the global parameters based on the cli arguments given.
    """
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

    for var, name in zip([N, TIMESPAN, DELTAK, DELTATAU],
                        ["N", "TIMESPAN", "DELTAK", "DELTATAU"]):
        print(f"{name}: {var}")


def printMatrix(matrix, name):
    """
    Prints the given matrix prettily.
    """
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
    """
    Returns the zero momentum state in the angular momentum basis.
    """
    state = np.zeros(DIM)
    state[N + 0] = 1
    return state

def densityOperator():
    """
    Returns the density operator of the initial state.
    """
    # states = uniformDistOnNSphere()
    # states = np.eye(DIM, SAMPLES) / SAMPLES
    states = np.array([zeroMomentumState()]).T
    # print("states.shape:", states.shape)
    rho = np.zeros((DIM, DIM))
    for i in range(SAMPLES):
        state = np.matrix(states[:, i]).T
        rho += state.dot(state.getH()) * (1/SAMPLES)
    return np.matrix(rho)

def findBesselBounds():
    """
    Finds the bounds on the v parameters of the Bessel jv
    function such that the absolute value of function
    jv(-K) is greater than EPSILON
    """
    v = 0
    while np.abs(jv(v, -K)) > EPSILON:
        v += 1

    return v

def denseFloquetOperator(t, deltak=0, deltatau=0):
    """
    Returns the Floquet operator in a dense form in order
    to allow optimising without issues with sparse matrices.
    [Sparse matrices have been removed in the recent versions
    of this code.]
    """
    tau = TAU + deltatau
    k = K + deltak
    n = np.arange(-N, N+1)
    colgrid, rowgrid = np.meshgrid(n, n)
    F = np.exp(-1j * BETA * np.sin(OMEGA * t * TAU) / (HBAR * OMEGA)) \
        * np.exp(-1j * tau * colgrid**2 / 2) \
        * jv(colgrid - rowgrid, -k) \
        * (1j)**(colgrid - rowgrid)

    return F

def floquetOperator(t, deltak=0, deltatau=0):
    """
    Returns the floquet operator in normal dense matrix form
    with entries less than EPSILON zeroed out.
    """
    F = denseFloquetOperator(t, deltak, deltatau)
    F[np.abs(F) < EPSILON] = 0
    # sparsity = 1 - np.count_nonzero(F)/(2*N + 1)**2
    # print("Sparsity: %.10f" % sparsity)
    # print("Total F Sum: %.10f" % np.sum(F))

    # F_sparse = csr_matrix(F)
    # return F_sparse

    return np.matrix(F)

def L2Operator():
    """
    Returns L^2 operator in matrix form in ang momentum basis.
    """
    L2 = HBAR**2 * np.diag(np.arange(-N, N+1)**2)
    # L2_sparse = csr_matrix(L2)
    # return L2_sparse
    return np.matrix(L2)

def ensembleAvg(rho, T):
    """
    Ensemble average of the T operator.
    Given by tr(T*rho).
    """
    # product = csr_matrix.dot(rho, T)
    # return ( product.diagonal(0) ).sum()
    return np.trace(T.dot(rho))

def Ldistribution(rho):
    """
    Input:
    -----
    rho: Density matrix of the state.

    Output:
    ------
    p: DIM length 1d array containing probabilities of L = m hbar.
       The probabilities correspond to m = arange(-N, N+1).
    """
    return np.diag(rho).real

def evolve(rho, F, Fh):
    """
    Evolves the density operator by one
    time period step.
    """
    return F.dot(np.dot(rho, Fh))

def getPeriodPerturbations(cumulative=True):
    """
    Returns a sequence of perturbations to the period of kicking.
    cumulative = True means we don't correct for the previous
    perturbation. This leads to a slow drift.
    cumulative = False means we correct for the previous perturbation
    in the next one.
    """
    period_perturbations = np.random.uniform(-DELTATAU, DELTATAU, TIMESPAN)
    if not cumulative:
        period_perturbations[1:] -= period_perturbations[:-1]
    return period_perturbations


def run():
    """
    Runs the simulations.

    Output:
    ------
    Averages of L^2 with time
    Variances of L^2 with time
    Probability distribution p(L = m*hbar) at the end.
    """
    # Both the operators are sparse, rho is not.
    rho = densityOperator()
    L2 = L2Operator()
    L4 = L2.dot(L2) # L^4 operator
    L2_ensemble_avgs = np.zeros(TIMESPAN, dtype=np.complex64)
    L2_ensemble_vars = np.zeros(TIMESPAN, dtype=np.complex64)

    if DELTAK != 0:
        kick_perturbations = np.random.uniform(-DELTAK, DELTAK, TIMESPAN)
    else:
        kick_perturbations = np.zeros(TIMESPAN)
    if DELTATAU != 0:
        period_perturbations = getPeriodPerturbations(cumulative=False)
    else:
        period_perturbations = np.zeros(TIMESPAN)

    for t in range(TIMESPAN):
        print(f"Starting cycle no {t}...")
        L2avg = ensembleAvg(rho, L2)
        L4avg = ensembleAvg(rho, L4)
        L2var = L4avg - L2avg**2

        L2_ensemble_avgs[t] = L2avg
        L2_ensemble_vars[t] = L2var

        F = floquetOperator(t, deltak=kick_perturbations[t],
            deltatau=period_perturbations[t])
        Fh = F.getH()
        rho = evolve(rho, F, Fh)

    probL = Ldistribution(rho)

    return L2_ensemble_avgs, L2_ensemble_vars, probL

def plot(avgs, varis, p):
    """
    Plots the data generated by running the simulation.
    Input:
    -----
    avgs: Averages of L^2 with time.
    varis: Variances of L^2 with time.
    p: Probability distribution p(L = m*hbar) at the end.
    """
    t = np.arange(TIMESPAN)
    n = np.arange(-N, N+1)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2,
        figsize=(16, 12))
    ax1.plot(t, avgs.real)
    ax1.set_yscale("log")
    ax1.set_ylabel(r"$[L^2]$")
    ax1.set_xlabel("t")

    ax3.plot(t, varis.real)
    # ax3.set_yscale("log")
    ax3.set_ylabel(r"$[\Delta(L^2)]$")
    ax3.set_xlabel("t")

    ax2.plot(n, np.log10(p))
    # ax2.scatter(n, p)
    # ax2.set_yscale("log")
    ax2.set_ylabel(r"$log_{10}(p(L = \hbar n))$")
    ax2.set_xlabel("n")
    title = f"Driven Kicked Quantum Rotor [K={K}, TAU={TAU}, DIM={DIM}," \
            f" T={TIMESPAN}, BETA={BETA}, OMEGA={OMEGA}]"
    fig.suptitle(title)
    plt.tight_layout()
    filename = f"plots/drivenquantumkickedrotor_T"\
                f"{TIMESPAN}DIM{DIM}BETA{BETA}OMEGA{OMEGA}"\
                f"_{date.today()}.png"
    plt.savefig(filename)

if __name__ == "__main__":
    initial_time = time.process_time()
    cmdArgSetter(argv)
    avgs, varis, p = run()
    plot(avgs, varis, p)
    final_time = time.process_time()
    print(f"This program took {final_time - initial_time}s of CPU time.")
    # print("p:", p)
    plt.show()
