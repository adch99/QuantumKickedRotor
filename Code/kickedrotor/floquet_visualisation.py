"""
Author: Aditya Chincholi
Purpose: To visualize the floquet matrix in terms of its absolute value
        phase structure.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import kickedrotor.quasiperiodic_rotor_3d as rotor

# F = np.random.normal(0, 1, (100, 100)) + 1j * np.random.normal(0, 1, (100, 100))
F = rotor.getFloquetOperator()
amps = np.abs(F)
phase = np.angle(F)

norm = colors.LogNorm(vmin=amps.min(), vmax=amps.max())


plt.figure()
mappable = plt.imshow(amps, norm=norm)
plt.colorbar(mappable)
plt.title("Amplitudes")

plt.figure()
mappable = plt.imshow(phase)
plt.title("Phases")
plt.colorbar(mappable)
plt.show()
