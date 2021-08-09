"""
A script that calculates the bipartite entanglement
entropy for a range of (K, alpha) values and then
looks at the trend across critical point.
"""

import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import seaborn as sns
import kickedrotor.bipartite_entanglement as rotor

# Parameters
omega2 = 2 * np.pi * np.sqrt(5)
omega3 = 2 * np.pi * np.sqrt(13)
hbar = 2.85
k_critical = 6.36
k_min = k_critical - 2
k_max = k_critical + 2
alpha_critical = 0.4303
alpha_min = alpha_critical - 0.2
alpha_max = alpha_critical + 0.2
samples = 11 # Must be odd
timesteps = 80

# Data files
save_data = True
save_plot = True
datafile_basename = "data/entanglement_multiK_multiAlpha"
plotfile_basename = "plots/entanglement_multiK_multiAlpha"



k_vals = np.linspace(k_min, k_max, samples)
# k_vals = np.zeros(samples) + k_critical
alpha_vals = np.linspace(alpha_min, alpha_max, samples)
# alpha_vals = np.zeros(samples) + alpha_critical
critical_index = (samples - 1) // 2

sns.set()
figs, axs = [], []
num_figs = 5
colours = sns.color_palette("icefire", samples)
colour_cycle = cycler(color=colours)
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

for ax in axs[:2]: ax.set_yscale("log")

energy_collection = np.array(energy_collection)
energy_diffs = np.diff(energy_collection, axis=0)
rotor.plotEnergyDiffs(energy_diffs, k_vals, timesteps, axs[3], critical_index)

entropy_collection = np.array(entropy_collection)
entropy_diffs = np.diff(entropy_collection, axis=0)
rotor.plotEntropyDiffs(entropy_diffs, k_vals, timesteps, axs[4], critical_index)

basenames = ["energies", "entropies", "momentum", "energy_diffs", "entropy_diffs"]

if save_plot:
    for i in range(num_figs):
        figs[i].tight_layout()
        filename = f"{plotfile_basename}_{basenames[i]}_N{rotor.N}_T{timesteps}"
        figs[i].savefig(filename+".pdf")
        figs[i].savefig(filename+".svg")

if save_data:
    filename = f"{datafile_basename}_N{rotor.N}_T{timesteps}.dat"
    datafile = open(filename, "w")
    datafile.write(f"# Timesteps: {timesteps} DIM: {rotor.DIM} Samples: {samples}\n")
    datafile.write("# K\n")
    np.savetxt(datafile, k_vals)
    datafile.write("\n# Alpha\n")
    np.savetxt(datafile, alpha_vals)
    datafile.write("\n# Energies\n")
    np.savetxt(datafile, energy_collection)
    datafile.write("\n# Entropies\n")
    np.savetxt(datafile, entropy_collection)



plt.show()
