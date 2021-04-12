#!/bin/python3

# Randomly Kicked Quantum Rotor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Physical Constants
k = 2
tau = 1
hbar = 1

# Program Constants
N = 10
DIM = 2*N + 1
TIMESTEPS = 100

def Fplus(state):
    multiplier = np.exp(-1j*k) * np.exp(-1j * tau * hbar * np.arange(-N, N+1)**2 / 2)
    return multiplier * state

def Fminus(state):
    multiplier = np.exp(1j*k) * np.exp(-1j * tau * hbar * np.arange(-N, N+1)**2 / 2)
    return multiplier * state

def getInitialState():
    real_state = np.random.normal(loc=0, scale=1, size=DIM)
    imag_state = np.random.normal(loc=0, scale=1, size=DIM)
    state = real_state + 1j*imag_state
    state = state / np.linalg.norm(state)
    return state

def run(initial_state, kick_sequence):
    state = initial_state
    stateHistory = [state]
    for t, kick in enumerate(kick_sequence):
        if kick == 1:
            state = Fplus(state)
        else:
            state = Fminus(state)

        stateHistory.append(state)
        print(f"At time t={t}, norm={np.conjugate(state).dot(state)}")
    return stateHistory

def thetaRepresentation(theta, momentumState):
    mtheta = np.tensordot(np.arange(-N,N+1), theta, axes=0)
    norm_const = np.sqrt(1/(2*np.pi))
    return norm_const * momentumState @ np.exp(1j * mtheta)

def plotStates(stateHistory):
    # Probability Distribution
    for t, state in enumerate(stateHistory):
        theta = np.linspace(0, 2*np.pi, 1000)
        theta_rep = thetaRepresentation(theta, state)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

        ax1.set_title(f"Theta Space State (t = {t})")
        ax1.set_xlabel(r"$\theta$")
        ax1.set_ylabel(r"$\psi (\theta)$")
        ax1.plot(theta, np.real(theta_rep), label=r"Re[$\psi(\theta)$]")
        ax1.plot(theta, np.imag(theta_rep), label=r"Im[$\psi(\theta)$]")
        ax1.set_xlim(0, 2*np.pi)
        ax1.set_ylim(-1.2, 1.2)

        p = np.real(np.conjugate(state) * state)
        m = np.arange(-N, N+1)
        ax2.set_title(f"Probability Distribution (t = {t})")
        ax2.set_xlabel("m")
        ax2.set_ylabel(r"$p(L = m \hbar)$")
        ax2.bar(m, p)

        phases = np.angle(state)
        rel_phases = phases - np.amin(phases)
        ax3.bar(m, rel_phases)
        ax3.set_ylim(0, 2*np.pi)
        ax3.set_title("Relative Phases")
        ax3.set_xlabel("m")
        ax3.set_ylabel("Relative Phase")

        plt.tight_layout()
        plt.savefig(f"plots/randomly_kicked_plots/t{t}.png")
        print(f"Outputting Image at t = {t}...")
        plt.close(fig)

def main():
    initial_state = getInitialState()
    kick_sequence = np.random.choice([-1, 1], size=TIMESTEPS, replace=True)
    stateHistory = run(initial_state, kick_sequence)
    plotStates(stateHistory)

if __name__ == "__main__":
    sns.set()
    main()
