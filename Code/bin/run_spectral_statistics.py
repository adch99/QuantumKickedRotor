#!python

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.linalg as linalg
import seaborn as sns
import kickedrotor.perturbed_quantum_kicked_rotor as rotor
# import kickedrotor.quasiperiodic_rotor_3d as rotor
# import kickedrotor.spectral_parity_decomposition as pdecomp
# import kickedrotor.params as params
# import kickedrotor.quantum_kicked_rotor_noparity as rotor
import kickedrotor.spectral_statistics as spectral
import kickedrotor.random_matrix_sampling as rmt


sns.set()
# N = 1000
# DIM = 2*N + 1


F = rotor.denseFloquetOperator()
# rmt.checkUnitarity(F)
F_type = "circular"
remove_degeneracy = True
deg_tol = 1e-8
abs_tol = 0.1

if F_type == "circular":
    phases, eigs, num_discard = spectral.getPhaseSpectrum(F, tol=abs_tol,
                                        discard=True, optimize=True)
    del F
    print(f"Number of Eigenphases in use = {phases.shape[0]}")
    print(f"Number of Discarded Phases = {num_discard}")

    if remove_degeneracy:
        phases_unique, uniqueness_ratio = spectral.removeDegeneracies(phases, deg_tol)
        print(f"Uniqueness ratio: {uniqueness_ratio}")
        spacings, ratios = spectral.getSpacingRatios(phases_unique)

        num_values = len(phases_unique)
        delta3, coeffs = spectral.getDelta3Statistic(phases_unique)

    else:
        spacings, ratios = spectral.getSpacingRatios(phases)
        delta3, coeffs = spectral.getDelta3Statistic(phases)
        num_values = len(phases)

elif F_type == "gaussian":
    eigvals = np.sort(linalg.eigvalsh(F))
    if remove_degeneracy:
        eigvals_unique, uniqueness_ratio = spectral.removeDegeneracies(eigvals, deg_tol)
        print(f"Uniqueness ratio: {uniqueness_ratio}")
        spacings, ratios = spectral.getSpacingRatios(eigvals_unique)

        num_values = len(eigvals_unique)
        delta3, coeffs = spectral.getDelta3Statistic(eigvals_unique)

    else:
        spacings, ratios = spectral.getSpacingRatios(eigvals)
        delta3, coeffs = spectral.getDelta3Statistic(eigvals_unique)
        num_values = len(eigvals)

else:
    print("Something is wrong. F_type should" \
        + " be either \"gaussian\" or \"circular\"")

num_infs = np.count_nonzero(np.isinf(ratios))
print(f"Discarding {num_infs} ratios as they are infs.")
np.nan_to_num(ratios, copy=False, nan=-1, posinf=-1, neginf=-1)

ratios_tilde = spectral.getSpacingRatiosTilde(ratios)
print(f"Mean spacing ratio = {np.mean(ratios)}")
print(f"Mean spacing ratio tilde = {np.mean(ratios_tilde)}")

print(f"Delta3 Statistic: {delta3}")
print(f"Observed Number of Eigenvalues: {num_values}")
expected_delta3_vals = spectral.meanDelta3(num_values)
print("Estimated Delta3 Poisson: %f \tGOE: %f \nGUE: %f \tGSE: %f" % expected_delta3_vals)


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4.5))
spectral.plotRatios(ratios, ax=ax1)
spectral.plotSpacings(spacings, ax=ax2)
# ax1.set_ylim(0, 1.1)
# ax2.set_ylim(0, 1.1)
# fig.suptitle(f"Quantum Kicked Rotor (K = {rotor.K})")
# fig.suptitle(f"Quasiperiodic Kicked Rotor (K = {params.K}, Alpha = {params.ALPHA})")
# plt.savefig(f"plots/quantum_kicked_rotor_N{rotor.N}_K{rotor.K}_spacing_ratios.pdf")
# plt.savefig(f"plots/quantum_kicked_rotor_N{rotor.N}_K{rotor.K}_spacing_ratios.svg")
plt.tight_layout()
# plt.savefig(f"plots/quasiperiodic_kickedrotor_spectrum_N{params.N}_K{params.K}_ALPHA{params.ALPHA}_HBAR{params.HBAR}.pdf")
# plt.savefig(f"plots/quasiperiodic_kickedrotor_spectrum_N{params.N}_K{params.K}_ALPHA{params.ALPHA}_HBAR{params.HBAR}.svg")
plt.show()
