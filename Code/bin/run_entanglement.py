"""
A script that calculates the bipartite entanglement
entropy for a range of (K, alpha) values and then
looks at the trend across critical point.
"""

import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import seaborn as sns
import kickedrotor.bipartite_entanglement_alt as rotor

def plotEnergyDiffs(energy_diffs, k_vals, timesteps, ax, critical_index):
    k_lowers = k_vals[:-1]
    t = np.arange(1, timesteps+1)
    for i, energy_diff in enumerate(energy_diffs):
        label = r"$K_l = $" + f"{k_lowers[i]:.3f}"
        linewidth = 1
        if i == critical_index: linewidth = 3
        ax.plot(t, energy_diff, label = label, marker=".",
            linewidth = linewidth)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$E_{K_{n+1}}(t) - E_{K_{n}}(t)$")
    ax.set_title("Energy Difference")
    # ax.legend()

def plotEntropyDiffs(entropy_diffs, k_vals, timesteps, ax, critical_index):
    k_lowers = k_vals[:-1]
    t = np.arange(1, timesteps+1)
    for i, entropy_diff in enumerate(entropy_diffs):
        linewidth = 1
        label = r"$K_l = $" + f"{k_lowers[i]:.3f}"
        if i == critical_index: linewidth = 3
        ax.plot(t, entropy_diff, label = label, marker = ".",
            linewidth = linewidth)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$S_{K_{n+1}}(t) - S_{K_{n}}(t)$")
    ax.set_yscale("log")
    ax.set_title("Entropy Difference")
    # ax.legend()

# Parameters
omega2 = 2*np.pi * np.sqrt(5)
omega3 = 2*np.pi * np.sqrt(13)
hbar = 2.85
k_critical = 6.36
k_min = k_critical - 3.36
k_max = k_critical + 3.36
alpha_critical = 0.4375
alpha_min = alpha_critical - 0.2375
alpha_max = alpha_critical + 0.2375
samples = 11 # Must be odd
timesteps = 80



k_vals = np.linspace(k_min, k_max, samples)
alpha_vals = np.linspace(alpha_min, alpha_max, samples)
critical_index = (samples - 1) // 2

sns.set()
figs, axs = [], []
num_figs = 5
colours = sns.color_palette("husl", 3)
colour_cycle = cycler(color=[colours[0]]*critical_index \
                            + [colours[1]] + [colours[2]]*critical_index)
lw_cycle = cycler(linewidth=[1]*critical_index \
                            + [3] + [1]*critical_index)
prop_cycle = colour_cycle + lw_cycle

for i in range(num_figs):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_prop_cycle(prop_cycle)
    figs.append(fig)
    axs.append(ax)

entropy_collection = []
energy_collection = []

for i in range(samples):
    initial_state = rotor.getInitialState()
    params = {
        "k" : k_vals[i],
        "alpha" : alpha_vals[i],
        "omega2" : omega2,
        "omega3" : omega3,
        "hbar" : hbar,
    }

    if i == critical_index:
        params["linewidth"] = 3
    print(f"Beginning K = {k_vals[i]:3f} alpha = {alpha_vals[i]:.3f}")
    result = rotor.run(initial_state, timesteps, params, matrix = False)
    final_state, entropies, energies, final_p1 = result
    entropy_collection.append(entropies)
    energy_collection.append(energies)
    rotor.plotEnergies(energies, ax = axs[0], save = False, **params)
    rotor.plotEntropies(entropies, ax = axs[1], save = False, **params)
    rotor.plotMomentum(final_p1, ax = axs[2], save = False, **params)

energy_collection = np.array(energy_collection)
energy_diffs = np.diff(energy_collection, axis=0)
plotEnergyDiffs(energy_diffs, k_vals, timesteps, axs[3], critical_index)

entropy_collection = np.array(entropy_collection)
entropy_diffs = np.diff(entropy_collection, axis=0)
plotEntropyDiffs(entropy_diffs, k_vals, timesteps, axs[4], critical_index)

basenames = ["energies", "entropies", "momentum", "energy_diffs", "entropy_diffs"]
for i in range(num_figs):
    figs[i].tight_layout()
    filename = f"plots/quasiperiodic_{basenames[i]}_N{rotor.N}_T{timesteps}_multiK"
    figs[i].savefig(filename+".pdf")
    figs[i].savefig(filename+".svg")

plt.show()
