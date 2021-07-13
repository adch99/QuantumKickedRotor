"""
This file contains the functions for an alternate
way of calculating the bipartite entanglement
entropy between the subspaces of the quasiperiodic
kicked rotor.
"""
import numpy as np
import scipy.fft as fft
import scipy.linalg as linalg
from scipy.special import xlogy

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Testing the floquet matrix
import kickedrotor.quasiperiodic_rotor_3d as matrix_generator

# Program Constants
N = 15
DIM = 2*N + 1
DTYPE = np.complex128
DIMF = DIM
TIMESTEPS = 10

# Scientific Constants
K = 3
ALPHA = 0.1
HBAR = 2.85
OMEGA2 = 2 * np.pi * np.sqrt(5)
OMEGA3 = 2 * np.pi * np.sqrt(13)

def getMomentumEvolution():
    """
    Returns a vector for the momentum evolution
    part of the floquet operator i.e.

    .. math:: U = exp(-i(p_1^2 / 2 + p_2 \\omega_2 + p_3 \\omega_3)/\\hbar)

    Returns
    --------
    momentum_phases : array_like
        A 1d array that should be elementwise multiplied
        with the state to give U |state>. `state` should
        be in the momentum basis.
    """
    momentum_phases = np.empty(DIM**3, dtype=DTYPE)
    for m1 in range(-N, N+1):
        for m2 in range(-N, N+1):
            for m3 in range(-N, N+1):
                m = DIM**2 * (m1 + N) + DIM * (m2 + N) + (m3 + N)
                phase = HBAR * m1**2 / 2 + m2 * OMEGA2 + m3 * OMEGA3
                momentum_phases[m] = np.exp(-1j * phase)
    return momentum_phases

def getAngleEvolution():
    """
    Returns a vector for the angular evolution
    part of the floquet operator i.e.

    .. math:: U = exp(-i K cos(\\theta_1) (1 + \\alpha cos(\\theta_2) cos(\\theta_3)) / \\hbar)

    Returns
    --------
    angle_phases : array_like
        A 1d array that should be elementwise multiplied
        with the state to give U |state>. `state` should
        be in the angular basis.
    """
    # I am assuming the state is represented by DIMF points [0, 2pi)
    # for each of the dimensions.

    angle_phases = np.empty(DIMF**3, dtype=DTYPE)
    for t1 in range(DIMF):
        for t2 in range(DIMF):
            for t3 in range(DIMF):
                t = DIMF**2 * t1 + DIMF * t2 + t3
                theta1 = 2 * np.pi * t1 / DIMF
                theta2 = 2 * np.pi * t2 / DIMF
                theta3 = 2 * np.pi * t3 / DIMF
                phase = (K / HBAR) * np.cos(theta1) \
                    * (1 + ALPHA * np.cos(theta2) * np.cos(theta3))
                angle_phases[t] = np.exp(-1j * phase)
    return angle_phases

def momentumToAngle(state):
    """
    Parameters
    -----------
    state : array_like
        1d array in momentum basis.

    Returns
    --------
    state_angle_basis : array_like
        Given state in angle basis.
    """
    reshaped_state = state.reshape(DIM, DIM, DIM)
    # reshaped_state = np.moveaxis(reshaped_state, [0, 1, 2], [1, 2, 0])
    params = {
        "x": fft.ifftshift(reshaped_state),
        "s": (DIMF, DIMF, DIMF),
        "norm": "ortho"
    }
    state_angle_basis = fft.ifftn(**params)
    return state_angle_basis.reshape(DIMF**3)

def angleToMomentum(state):
    """
    Parameters
    -----------
    state : array_like
            1d array in angle basis.

    Returns
    --------
    state_momentum_basis : array_like
        Given state in momentum basis.
    """
    reshaped_state = state.reshape(DIMF, DIMF, DIMF)
    params = {
        "x": reshaped_state,
        "s": (DIM, DIM, DIM),
        "norm": "ortho"
    }
    state_momentum_basis = fft.fftshift(fft.fftn(**params))
    # state_momentum_basis = np.moveaxis(state_momentum_basis, [0, 1, 2], [-1, 0, 1])
    return state_momentum_basis.reshape(DIM**3)

def performTimeEvolution(state, momentum_phases, angle_phases):
    """
    Returns the time evolved version of the given
    state corresponding to time evolution by one
    time step i.e. F|psi>.
    The state should be in the momentum basis and
    the returned state will be in the momentum basis
    as well.

    Parameters
    -----------
    state : array_like
        1d array of the state vector in the momentum
        basis.
    momentum_phases : array_like
        Diagonal of the momentum part of the floquet
        operator in momentum basis.
    angle_phases : array_like
        Diagonal of the angular part of the floquet
        operator in angle basis.

    Returns
    --------
    new_state : array_like
        1d array of the new state vector after time
        evolving `state` by 1 timestep i.e. F|state>.
    """
    new_state = momentum_phases * state
    new_state_angle_basis = momentumToAngle(new_state)
    new_state_angle_basis = angle_phases * new_state_angle_basis
    new_state = angleToMomentum(new_state_angle_basis)
    return new_state

def performTimeEvolutionMatrix(state, floquet_matrix):
    """
    Returns the time evolved version of the given
    state corresponding to time evolution by one
    time step i.e. F|psi>.
    The state should be in the momentum basis and
    the returned state will be in the momentum basis
    as well.
    This differs from `performTimeEvolution` as it
    applies the evolution in two parts: apply the
    diagonal momentum evolution, iFFT to change basis,
    diagonal angular evolution, FFT to change basis back.
    This function just dots the state with the floquet_matrix.

    Parameters
    -----------
    state : array_like
        1d array of the state vector in the momentum
        basis.

    Returns
    --------
    new_state : array_like
        1d array of the new state vector after time
        evolving `state` by 1 timestep i.e. F|state>.
    """
    new_state = np.dot(floquet_matrix, state)
    return new_state


def getDensityMatrix(state):
    """
    Returns the density matrix of the system corresponding
    to the pure state.

    Parameters
    ----------
    state : array_like
        State vector of the system in momentum basis.

    Returns
    -------
    rho : array_like
        Density matrix of the system corresponding to |state><state|.
    """
    return np.outer(state, state.conjugate())


def partialTrace(rho):
    """
    Returns partial trace over theta2+theta3 space.

    Parameters
    -----------
    rho : array_like
        2d array representing the density matrix in the
        momentum basis.

    Returns
    --------
    rho1 : array_like
        2d array representing the reduced density matrix
        corresponding to p1 in momentum basis.
    """
    rho_product = rho.reshape((DIM,)*6)
    return np.einsum("ijkljk", rho_product)

def reducedDensityMatrix(state):
    """
    Returns the reduced density matrix of the theta1 subspace.
    Equivalent to creating the overall density matrix and then
    taking a partial trace over theta2+theta3 subspace.

    Parameters
    ----------
    state : array_like
        1d array representing the state of the system
        in momentum basis (though the basis should not
        matter).

    Returns
    -------
    rho1 : array_like
        2d array representing the reduced density matrix
        of the given state for the theta1 subspace.
    """
    state_reshaped = state.reshape((DIM,)*3)
    return np.einsum("ikl,jkl->ij", state_reshaped, state_reshaped.conj())


def vonNeumannEntropy(rho1):
    """
    Returns the von Neumann entropy of the given density matrix.

    Parameters
    ----------
    rho1 : array_like
        Reduced density matrix after taking partial trace over
        dimensions 2 and 3.
    """
    eigvals = linalg.eigvals(rho1)
    return np.sum(-xlogy(eigvals, eigvals))

def getEnergy():
    """
    Returns a vector for the getting the expectation value of the energy.

    .. math: sum_{\mathbf{p}} \frac{p_1^2}{2} + p_2 \omega_2 + p_3 \omega_3

    Returns
    --------
    energy_values : array_like
        A 1d array that should be dotted with the state to
        give the energy of the state. `state` should be in
        the momentum basis.
    """
    energy_values = np.empty(DIM**3, dtype=DTYPE)
    for m1 in range(-N, N+1):
        for m2 in range(-N, N+1):
            for m3 in range(-N, N+1):
                m = DIM**2 * (m1 + N) + DIM * (m2 + N) + (m3 + N)
                energy_values[m] = HBAR * m1**2 / 2 + m2 * OMEGA2 + m3 * OMEGA3
    return energy_values

def getInitialState():
    """
    Returns the initial state of the simulation.

    Returns
    -------
    state : array_like
        1d array representing the initial state in the
        momentum cross-product basis. Currently set to
        |0> x |S> x |S> where |S> is the uniform
        superposition of all the states.
    """
    state1 = np.zeros(DIM)
    state1[N] = 1
    state23 = np.ones(DIM) / np.sqrt(DIM)
    state = np.kron(state1, np.kron(state23, state23))
    return state

def plotEntropies(entropies, ax = None, save = True):
    if ax is None:
        ax = plt.gca()
    time = np.arange(1, TIMESTEPS+1)
    ax.plot(time, entropies, marker="o", label=r"$\alpha = $"+f"{ALPHA:.2f}")
    ax.set_xlabel("t")
    ax.set_ylabel("Entanglement Entropy")
    ax.set_title("Bipartite Entanglement Entropy in Quasiperiodic Kicked Rotor")
    ax.legend()

    if save:
        plt.tight_layout()
        plt.savefig(f"plots/quasiperiodic_entropies_N{N}_ALPHA{ALPHA:.3f}.pdf")
        plt.savefig(f"plots/quasiperiodic_entropies_N{N}_ALPHA{ALPHA:.3f}.svg")

def plotEnergies(energies, ax = None, save = True):
    if ax is None:
        ax = plt.gca()
    time = np.arange(1, TIMESTEPS+1)
    ax.plot(time, energies, marker="o", label=r"$\alpha = $"+f"{ALPHA:.2f}")
    ax.set_xlabel("t")
    ax.set_ylabel("Energy")
    ax.set_title("Energy Evolution of Quasiperiodic Kicked Rotor")
    ax.legend()

    if save:
        plt.tight_layout()
        plt.savefig(f"plots/quasiperiodic_energies_N{N}_ALPHA{ALPHA:.3f}.pdf")
        plt.savefig(f"plots/quasiperiodic_energies_N{N}_ALPHA{ALPHA:.3f}.svg")


def plotMomentum(momentum, ax = None, save = True):
    if ax is None:
        ax = plt.gca()
    p = np.arange(-N, N+1)
    ax.plot(p, momentum, marker="o", label=r"$\alpha = $"+f"{ALPHA:.2f}")
    ax.set_xlabel(r"$\frac{p}{\hbar}$")
    ax.set_ylabel("P(p)")
    ax.set_yscale("log")
    ax.set_title("Momentum Distribution of Final State")
    ax.legend()

    if save:
        plt.tight_layout()
        plt.savefig(f"plots/quasiperiodic_momenta_N{N}_ALPHA{ALPHA:.3f}.pdf")
        plt.savefig(f"plots/quasiperiodic_momenta_N{N}_ALPHA{ALPHA:.3f}.svg")

def run(initial_state, timesteps, matrix = False):
    state = initial_state

    if not matrix: # Main and scalable way of time evolution
        momentum_phases = getMomentumEvolution()
        angle_phases = getAngleEvolution()

    else: # Direct matrix method
        floquet_matrix = matrix_generator.getFloquetOperator()

    energy_values = getEnergy()
    # rho = np.empty((DIM**3, DIM**3), dtype=DTYPE)
    rho1 = np.empty((DIM, DIM), dtype=DTYPE)
    entropies = np.empty(timesteps)
    energies = np.empty(timesteps)
    for t in range(timesteps):
        if not matrix: # Main way
            state = performTimeEvolution(state, momentum_phases, angle_phases)
        else: # Direct matrix method
            state = performTimeEvolutionMatrix(state, floquet_matrix)

        # rho[:, :] = getDensityMatrix(state)
        # print(f"Trace of rho = {np.trace(rho):.3f}")
        # rho1[:, :] = partialTrace(rho)

        rho1 = reducedDensityMatrix(state)
        print(f"Trace of rho1 = {np.trace(rho1):.3f}")
        entropies[t] = vonNeumannEntropy(rho1).real
        print(f"At time step {t+1}, we have entropy: {entropies[t]:.3f}")
        energies[t] = np.dot(energy_values, np.abs(state)**2)
        print(f"At time step {t+1}, we have energy: {energies[t]:.3f}")

    final_p1 = np.diag(rho1)

    return state, entropies, energies, final_p1

def compareStates(state1, state2, atol = 1e-8, rtol = 1e-5):
    are_close = np.allclose(state1, state2, atol = atol, rtol = rtol)
    match_number = np.count_nonzero(np.isclose(state1, state2, atol=atol, rtol=rtol))
    match_percent = 100 * match_number / DIM**3
    diff = np.abs(state1 - state2)
    max_diff = diff.max()
    mean_diff = np.mean(diff)
    if are_close: print("Close")
    else: print("Not Close")

    return max_diff, mean_diff, match_number, match_percent


def compareEvolutions(state, floquet_matrix, momentum_phases, angle_phases,
    atol = 1e-8, rtol = 1e-5):
    exp_state = performTimeEvolution(state, momentum_phases, angle_phases)
    obs_state = performTimeEvolutionMatrix(state, floquet_matrix)

    (max_diff, mean_diff,
        match_number, match_percent) = compareStates(exp_state, obs_state,
                                                    atol=atol, rtol=rtol)
    # print(f"They are different by max = {diff.max():.3e}")
    # print(f"The mean difference is {np.mean(diff):.3e}")
    # print(f"The mean relative difference is {np.mean(diff / np.abs(exp_state)):.3e}")
    # print(f"The other mean relative difference is {np.mean(diff / np.abs(obs_state)):.3e}")
    # print(f"The number of matches is {match_number}/{DIM**3} i.e. {match_percent:.2f}%")
    # print()
    return max_diff, mean_diff, match_number, match_percent

def main():
    initial_state = getInitialState()
    final_state, entropies, energies, final_p1 = run(initial_state, TIMESTEPS,
                                                    matrix = True)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 4.5))
    plotEntropies(entropies, ax = ax[0, 0], save = False)
    plotEnergies(energies, ax = ax[0, 1], save = False)
    plotMomentum(final_p1, ax = ax[1, 1], save = False)

    plt.tight_layout()
    # plt.savefig(f"plots/quasiperiodic_plots_N{N}_ALPHA{ALPHA:.3f}_K{K:.1f}.pdf")
    # plt.savefig(f"plots/quasiperiodic_plots_N{N}_ALPHA{ALPHA:.3f}_K{K:.1f}.svg")

if __name__ == "__main__":
    sns.set()
    main()
    plt.close()
    plt.show()
