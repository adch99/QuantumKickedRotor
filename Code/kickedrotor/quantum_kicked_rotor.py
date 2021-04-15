import numpy as np
from scipy.special import jv # Bessel function of first kind
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs as speigs
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
sns.set()

# Physical Constants
# hbar = 1.0545718e-34
hbar = 2.85
tau = 1 * hbar
k = 5 / hbar

# Program Constants
N = 1000
DIM = 2*N + 1 # [-N, N]
EPSILON = 1e-6
SAMPLES = 1
TIMESPAN = 1000


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
    # print("new_norms:", np.linalg.norm(bases/norms, axis=0))
    return bases / norms
    # x = np.matrix(np.zeros(DIM)).T
    # x[N+5] = 1
    # return x

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

def floquetOperator():
    """
    Returns the floquet operator in sparse matrix form.
    """
    F = np.zeros((DIM, DIM), dtype=np.complex64)

    for n in range(-N, N+1):
        for m in range(-N, N+1):
            F[N+m, N+n] = np.exp(-1j * tau * n**2 / 2) * jv(n-m, -k) * (1j)**(n-m)

    F[np.abs(F) < EPSILON] = 0
    # print(f"Max: {np.max(np.abs(F))}")
    # sign, logdetF = np.linalg.slogdet(F)
    # detF = sign * np.exp(logdetF)
    # print(f"|F| = {detF}\t sign = {sign}\t logdetF = {logdetF}")
    # print(F)
    # F /= detF
    sparsity = 1 - np.count_nonzero(F)/(2*N + 1)**2
    # print("Sparsity: %.10f" % sparsity)
    # print("Total F Sum: %.10f" % np.sum(F))
    F_sparse = csr_matrix(F)
    return F_sparse

def standardiseForIFFT(v):
    """
    It turns out IFFT needs the frequencies in a specific
    order: 0 1 2 3 .... N -1 -2 -3 ... -N
    But our frequency spectrum is like:
    -N ... -2 -1 0 1 2 ... N
    So we 'standardise' it for the sake of IFFT.
    """
    stdv = np.zeros(DIM, dtype=np.complex64)
    stdv[:N+1] = v[N:].flatten()
    stdv[N+1:] = v[:N][::-1].flatten()
    return stdv

# def thetaSpaceTransform(v):
    # return lambda theta: np.sum(v*np.exp(1j*np.arange(-N, N)*))

def visualizeFloquetEigs(F):
    eigvals, eigvecs = np.linalg.eig(F.todense())
    print(eigvals)

    numvecs = min(len(eigvecs), 6)
    sortindex = np.argsort(eigvals)[::-1]

    fig, ax = plt.subplots(nrows=numvecs, figsize=(16, 6*numvecs))
    numpts = 1000
    theta = np.matrix(np.linspace(0, 2*np.pi, numpts))
    freq = np.matrix(np.arange(-N, N+1)).T
    for i in range(numvecs):
        index = sortindex[i]
        eigvec = eigvecs[:, index]
        exponents = np.array(1j * freq @ theta)
        wavefunc = (eigvec.T @ np.matrix(np.exp(exponents))).T

        # print("eigvec.shape:", eigvec.shape)
        # print("theta.shape:", theta.shape)
        # print("exponents.shape:", exponents.shape)
        # print("wavefunc.shape:", wavefunc.shape)

        ax[i].set_title(f"Eigenvector {i} with eigenvalue {eigvals[index]}")
        ax[i].plot(theta.T, wavefunc.real, label="real", alpha=0.9)
        ax[i].plot(theta.T, wavefunc.imag, label="imag", alpha=0.9)
        ax[i].legend()
    plt.savefig("plots/floqueteigenvecs.png")

def L2Operator():
    L2 = hbar**2 * np.diag(np.arange(-N, N+1)**2)
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
    # print("rho:")
    # print(rho)
    for n in range(-N, N+1):
        basis_vector = np.zeros(DIM)
        basis_vector[n+N] = 1
        basis_vector = csr_matrix(basis_vector).T
        # print(basis_vector)
        p[n+N] = (basis_vector.T.dot(csr_matrix.dot(rho, basis_vector))).item().real
        # print(p[n+N])
    return p

def evolve(rho, F, Fh):
    """
    Evolves the density operator by one
    time period step.
    """
    return F.dot(csr_matrix.dot(rho, Fh))

def run():
    # Both the operators are sparse, rho is not.
    rho = densityOperator()
    F = floquetOperator()
    Fh = F.getH() # Hermitian of F
    # with np.printoptions(precision=2, suppress=True):
    #     print(F.dot(Fh).todense())
    L2 = L2Operator()
    L4 = L2.dot(L2) # L^4 operator
    L2ensembleavgs = np.zeros(TIMESPAN, dtype=np.complex64)
    L2ensemblevars = np.zeros(TIMESPAN, dtype=np.complex64)

    p = Ldistribution(rho)
    # print("Starting dist:", p)
    # print("Starting rho:", rho)
    # L2avg = EnsembleAvg(rho, L2)
    # L4avg = EnsembleAvg(rho, L4)
    # L2var = L4avg - L2avg**2
    # print(f"L2avg: {L2avg}\t L4avg: {L4avg}\t L2var: {L2var}")

    for t in range(TIMESPAN):
        print(f"Starting cycle no {t}...")
        # rhotrace = np.trace(rho)
        # if 1 - EPSILON < rhotrace and rhotrace > 1 + EPSILON:
        #     print(f"Trace(rho): {np.trace(rho)}")
        L2avg = EnsembleAvg(rho, L2)
        L4avg = EnsembleAvg(rho, L4)
        L2var = L4avg - L2avg**2

        L2ensembleavgs[t] = L2avg
        L2ensemblevars[t] = L2var

        rho = evolve(rho, F, Fh)
        # if np.isnan(rho).any():
        #     print("rho has NaN at:", np.where(np.isnan(rho)))
        #     # print("rho:", rho)
        #     exit()

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
    ax3.set_yscale("log")
    ax3.set_ylabel(r"$[\Delta(L^2)]$")
    ax3.set_xlabel("t")

    ax2.plot(n, np.log10(p))
    # ax2.scatter(n, p)
    # ax2.set_yscale("log")
    ax2.set_ylabel(r"$log_{10}(p(L = \hbar n))$")
    ax2.set_xlabel("n")

    fig.suptitle(f"Kicked Quantum Rotor [k={k}, tau={tau}, DIM={DIM}, T={TIMESPAN}]")
    plt.tight_layout()
    plt.savefig(f"plots/quantumkickedrotor_T{TIMESPAN}DIM{DIM}_{date.today()}.png")

if __name__ == "__main__":
    avgs, vars, p = run()
    plot(avgs, vars, p)
    # print("p:", p)
    plt.show()
