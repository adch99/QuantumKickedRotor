import numpy as np
import matplotlib.pyplot as plt

import kickedrotor.bipartite_entanglement_alt as rotor
import kickedrotor.quasiperiodic_rotor_3d as matrix_generator
from kickedrotor.params import *


# floquet_matrix = matrix_generator.getFloquetOperator()
# np.save(f"floquet_matrix_{N}.npy", floquet_matrix)
# exit(0)

momentum_phases = rotor.getMomentumEvolution()
angle_phases = rotor.getAngleEvolution()
floquet_matrix = np.lib.format.open_memmap(f"floquet_matrix_{N}.npy",
    dtype = np.complex128,
    mode = "r",
    shape = (DIM**3, DIM**3)
    )
# exit()

midpoint = (DIM**3) // 2
indices = np.arange(0, DIM**3)
n = len(indices)
max_diffs = np.zeros(n)
mean_diffs = np.zeros(n)
match_percents = np.zeros(n)
match_numbers = np.zeros(n)

for i in range(n):
    index = indices[i]
    initial_state = np.zeros((DIM**3,))
    initial_state[index] = 1
    print(f"Index {index}...")
    # print("Running direct method... ", end="", flush=True)
    direct_state = rotor.performTimeEvolution(initial_state,
                    momentum_phases, angle_phases)
    # print("Done")
    # print("Max(direct_state):", np.abs(direct_state).max())

    matrix_state = floquet_matrix[:, index]
    # print("Max(matrix_state):", np.abs(matrix_state).max())
    args = (direct_state, matrix_state)
    kwargs = {"atol": 1e-5, "rtol": 1e-8}
    comparison = rotor.compareStates(*args, **kwargs)
    (max_diffs[i], mean_diffs[i],
        match_numbers[i], match_percents[i]) = comparison
    print()

titles = ["Max Diff", "Mean Diff", "Match %", "Unmatched"]
ylabels = ["max diff", "mean diff", "match %", "unmatched elements"]
ydata = [max_diffs, mean_diffs, match_percents, (DIM**3 - match_numbers)]
filenames = ["max_diff", "mean_diff", "match_percent", "unmatched"]


# Plot everything
for i in range(len(titles)):
    fig = plt.figure()
    plt.subplot(121)
    plt.scatter(indices, ydata[i], marker="o")
    # plt.xticks(indices[::5])
    plt.xlabel("index")
    plt.ylabel(ylabels[i])
    fig.suptitle(titles[i])

    if "Diff" in titles[i]:
        plt.yscale("log")

    plt.subplot(122)
    if "Diff" in titles[i]:
        bins = np.logspace(np.log10(ydata[i].min())-1, 1+np.log10(ydata[i].max()), 10)
        plt.xscale("log")
    else:
        bins = 10
    plt.hist(ydata[i], bins=bins)
    plt.xlabel(ylabels[i])

    plt.tight_layout()
    fname = f"plots/quasiperiodic_matrix_N{N}_" + filenames[i] + ".pdf"
    plt.savefig(fname)

plt.show()
