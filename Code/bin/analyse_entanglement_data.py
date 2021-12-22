#!/bin/python

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

datafilename = "data/entanglement_multiK_multiAlpha_power2_N128_T250.npz"
data = np.load(datafilename)
samples, timesteps = data["energy_collection"].shape

fig, ax = plt.subplots(figsize=(8, 6))


t = np.arange(1, timesteps+1)
for i in range(samples):
    label = f"K = {data['k_vals'][i]:.3f} α = {data['alpha_vals'][i]:.3f}"
    colour = next(ax._get_lines.prop_cycler)["color"]
    x = np.log10(t)
    y = np.log10(data["energy_collection"][i])

    slope, intercept = np.polyfit(x, y, deg=1)
    xfit = np.linspace(x.min(), x.max(), 100)
    print(f"{label}\nslope = 2/3 * {slope*3/2:.3f}\n")


    ax.plot(x, y, label=label, color=colour)
    ax.plot(xfit, slope*xfit + intercept, linestyle="--",
            color=colour, label = f"y = {slope:.2f}x + {intercept:.2f}")

ax.set_xlabel(r"$log_{10} t$")
ax.set_ylabel(r"$log_{10} E$")
ax.set_title("Energies")
# ax.set_xscale("log")
# ax.set_yscale("log")
ax.legend(frameon=False)

# fig.show()
fig.savefig("plots/momentum_growth_verification_N128_T250.pdf")
plt.show()

# # Output
# K = 4.360 α = 0.230
# slope = 2/3 * 0.044
#
# K = 5.360 α = 0.330
# slope = 2/3 * 1.153
#
# K = 6.360 α = 0.430
# slope = 2/3 * 1.109
#
# K = 7.360 α = 0.530
# slope = 2/3 * 1.535
#
# K = 8.360 α = 0.630
# slope = 2/3 * 1.544
