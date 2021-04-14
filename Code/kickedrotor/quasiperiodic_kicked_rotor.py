from sys import argv
from datetime import date
import time
import numpy as np
from scipy.special import jv # Bessel function of first kind
# from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Physical Constants
# HBAR = 1.0545718e-34
HBAR = 1
TAU = 1
ALPHA = 0 # Quasiperiodicity constant
OMEGA2 = 2 * np.pi * np.sqrt(5)
OMEGA3 = 2 * np.pi * np.sqrt(13)
KMIN = 5
KMAX = 7
DELTAK = 0 # 1e-2 / HBAR
DELTATAU = 0 #1e-2 / HBAR

# Program Constants
N = 1000
DIM = 2*N + 1 # [-N, N]
EPSILON = 1e-9
SAMPLES = 1
KSAMPLES = 5
TIMESPAN = 200
TMIN = 30

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
    Prints the given matrix prettily. For debugging purposes.
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


def denseFloquetOperator(t, base_strength, deltak=0, deltatau=0):
    """
    Returns the Floquet operator in a dense form in order
    to allow optimising without issues with sparse matrices.
    [Sparse matrices have been removed in the recent versions
    of this code.]
    """
    tau = TAU + deltatau
    k = base_strength + deltak

    # Kick strength is k (1 + α cos(ω_2 t τ) cos(ω3 t τ))
    kick_strength = k * (1 + ALPHA * np.cos(OMEGA2 * t * tau) * np.cos(OMEGA3 * t * tau))
    n = np.arange(-N, N+1)
    colgrid, rowgrid = np.meshgrid(n, n)
    F = np.exp(-1j * tau * colgrid**2 / (2*HBAR)) \
        * jv(colgrid - rowgrid, -kick_strength / HBAR) \
        * (1j)**(colgrid - rowgrid)
    return F

def floquetOperator(t, base_strength, deltak=0, deltatau=0):
    """
    Returns the floquet operator in normal dense matrix form
    with entries less than EPSILON zeroed out.
    """
    F = denseFloquetOperator(t, base_strength, deltak, deltatau)
    F[np.abs(F) < EPSILON] = 0
    return np.matrix(F)

def L2Operator():
    """
    Returns L^2 operator in matrix form in ang momentum basis.
    """
    L2 = HBAR**2 * np.diag(np.arange(-N, N+1)**2)
    return np.matrix(L2)

def ensembleAvg(rho, T):
    """
    Ensemble average of the T operator.
    Given by tr(T*rho).
    """
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


def run(k):
    """
    Runs the simulation for a particular k value.

    Output:
    ------
    """
    print(f"Starting run for k = {k}...")

    rho = densityOperator()
    L2 = L2Operator()
    L2_ensemble_avgs = np.zeros(TIMESPAN, dtype=np.complex64)

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
        L2_ensemble_avgs[t] = L2avg

        F = floquetOperator(t+1, k, deltak=kick_perturbations[t],
            deltatau=period_perturbations[t])
        Fh = F.getH()
        rho = evolve(rho, F, Fh)

    probL = Ldistribution(rho)
    time = np.arange(0, TIMESPAN)
    log_lambdas = np.log(L2_ensemble_avgs * time**(-2/3) / HBAR**2)

    plotAvg(L2_ensemble_avgs, probL, k)

    return log_lambdas

def plotAvg(avgs, probL, k):
    t = np.arange(TIMESPAN)
    m = np.arange(-N, N+1)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.plot(t, avgs, label="k")
    ax1.set_title(f"K = {k} hbar={HBAR}")
    ax1.set_xlabel("t")
    ax1.set_ylabel(r"$[L^2]$")

    ax2.plot(m, probL, label="k")
    ax2.set_title(f"K = {k}")
    ax2.set_xlabel("m")
    ax2.set_ylabel(r"$p(L = m \hbar)$")
    ax2.set_yscale("log")


def main():
    kvalues = np.linspace(KMIN, KMAX, KSAMPLES)
    log_lambda_collection = np.empty((KSAMPLES, TIMESPAN))
    for i, k in enumerate(kvalues):
        log_lambda_collection[i,:] = run(k)

    plot(log_lambda_collection)


def plot(log_lambda_collection):
    """
    Plots the data generated by running the simulation.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1)

    k = np.linspace(KMIN, KMAX, KSAMPLES)
    for t, log_lambdas in enumerate(log_lambda_collection.T):
        if t > TMIN:
            ax.scatter(k, log_lambdas, label=f"{t}")
            ax.plot(k, log_lambdas, linestyle="--", alpha=0.6)

    title = f"Quasiperiodic Quantum Kicked Rotor [TAU={TAU}, DIM={DIM}," \
            f" T={TIMESPAN}]"
    ax.set_title(title)
    ax.set_ylabel(r"$ln \Lambda$")
    ax.set_xlabel("K")
    # plt.legend()

    filename = f"plots/quasiperiodic_T{TIMESPAN}DIM{DIM}_{date.today()}.png"
    plt.savefig(filename)


if __name__ == "__main__":
    sns.set()
    initial_time = time.process_time()
    cmdArgSetter(argv)
    # main()
    run(5)
    final_time = time.process_time()
    print(f"This program took {final_time - initial_time}s of CPU time.")
    # print("p:", p)
    plt.show()
