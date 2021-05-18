#!python

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kickedrotor.perturbed_quantum_kicked_rotor as rotor
import kickedrotor.spectral_statistics as spectral

sns.set()

F = rotor.denseFloquetOperator(deltak=0, deltatau=0)
phases, num_discard = spectral.getPhaseSpectrum(F, tol=1e-9, discard=True)
print(f"Number of Eigenphases in use = {phases.shape[0]}")
print(f"Number of Discarded Phases = {num_discard}")
spacings, ratios = spectral.getSpacingRatios(phases)
num_infs = np.count_nonzero(np.isinf(ratios))
print(f"Discarding {num_infs} ratios as they are infs.")
np.nan_to_num(ratios, copy=False, nan=-1, posinf=-1, neginf=-1)

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
spectral.plotRatios(ratios, ax=ax1)
spectral.plotSpacings(spacings, ax=ax2)
plt.savefig("plots/kickedrotor_spectrum.png")
plt.show()
