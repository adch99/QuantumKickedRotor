#!python

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import scipy.linalg as linalg
import seaborn as sns
import kickedrotor.perturbed_quantum_kicked_rotor as rotor
# import kickedrotor.quasiperiodic_rotor_3d as rotor
import kickedrotor.spectral_statistics as spectral
# import kickedrotor.params as params
sns.set()

F = rotor.denseFloquetOperator()
# F = rotor.getFloquetOperator()
# F = stats.unitary_group.rvs(1000)
phases, eigvals, num_discard = spectral.getPhaseSpectrum(F, tol=0.1, discard=True)
# phases *= 1 / params.HBAR
print(f"Number of Eigenphases in use = {phases.shape[0]}")
print(f"Number of Discarded Phases = {num_discard}")
spacings, ratios = spectral.getSpacingRatios(phases)
plt.hist(spacings, bins=100)
plt.show()
plt.close()
num_infs = np.count_nonzero(np.isinf(ratios))
print(f"Discarding {num_infs} ratios as they are infs.")
np.nan_to_num(ratios, copy=False, nan=-1, posinf=-1, neginf=-1)

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
spectral.plotRatios(ratios, ax=ax1)
spectral.plotSpacings(spacings, ax=ax2)
plt.savefig(f"plots/kickedrotor_spectrum.pdf")
# plt.savefig(f"plots/quasiperiodic_kickedrotor_spectrum_N{params.N}_K{params.K}_ALPHA{params.ALPHA}_HBAR{params.HBAR}.pdf")
plt.show()
