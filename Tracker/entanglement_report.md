---
title: Report on Bipartite Entanglement Entropy
author: Aditya Chincholi
date: June 28, 2021
theme: metropolis
aspectratio: 169
bibliography: citations.bib
csl: nature
...

# Introduction

## System

$$
    H = \frac{p_1^2}{2} + p_2 \omega_2 + p_3 \omega_3 +
    K cos(\theta_1) ( 1 + \alpha cos(\theta_2) cos(\theta_3))
    \sum_n \delta(t - n)
$$

- We are calculating the entanglement entropy between $\mathcal{H}_1$ and
    $\mathcal{H}_{23}$.

- The hypothesis was that the entanglement should increase rapidly at the
    critical point.

- Calculated and plotted quantities are:
    1. Entanglement entropy: $S = -Tr(\rho_1 ln(\rho_1))$
    2. Energy: $E = p_1^2 / 2 + p_2 \omega_2 + p_3 \omega_3$
    3. Momentum: $P(p_1 = m\hbar)$
    4. Entropy Change: $S_{K_{n+1}, \alpha_{n+1}} - S_{K_n, \alpha_n}$
    5. Energy Change: $E_{K_{n+1}, \alpha_{n+1}} - E_{K_n, \alpha_n}$

------------------------------------------------------------------------------

## Parameters

- We use $\hbar = 1$, $\omega_2 = 2\pi\sqrt{5}$ and
    $\omega_3 = 2\pi\sqrt{13}$. The critical point of the metal
    insulator transition is known to occur at $K_c = 6.36 \pm 0.02$.

- The value of $\alpha_c$, however is not determined accurately,
    nor written in any literature as it doesn't form a part of the
    scaling function in a nice way, though it most certainly does
    matter as it generates the anisotropy. We determine
    the value of $\alpha_c$ roughly by using the fact
    that the authors of the cited papers traversed the
    $K-\alpha$ space along a straight line perpendicular
    to the transition parabola. [@lemarieUniversalityAndersonTransition2009]
    [@lemarieObservationAndersonMetalinsulator2009]

- TL;DR we used $K_c = 6.36$ and $\alpha_c = 0.4303$.

- We used a basis size of 201 (-100 to 100) for each of the
    3 coordinates and the simulations were done for 80 timesteps.


# Varying $K$

----------------------------------------------------------------------------

![Entropy][multiK_entropies]

-----------------------------------------------------------------------------

![Energy][multiK_energies]

-----------------------------------------------------------------------------

![Momentum][multiK_momentum]

-----------------------------------------------------------------------------

![Entropy Increase with K][multiK_entropy_diffs]

-----------------------------------------------------------------------------

![Energy Increase with K][multiK_energy_diffs]

[multiK_entropies]: img/20210728/entanglement_multiK_entropies.pdf { .center height=98% }
[multiK_energies]: img/20210728/entanglement_multiK_energies.pdf { .center height=98% }
[multiK_momentum]: img/20210728/entanglement_multiK_momentum.pdf { .center height=98% }
[multiK_entropy_diffs]: img/20210728/entanglement_multiK_entropy_diffs.pdf { .center height=98% }
[multiK_energy_diffs]: img/20210728/entanglement_multiK_energy_diffs.pdf { .center height=98% }

# Varying $\alpha$

----------------------------------------------------------------------------

![Entropy][multiAlpha_entropies]

-----------------------------------------------------------------------------

![Energy][multiAlpha_energies]

-----------------------------------------------------------------------------

![Momentum][multiAlpha_momentum]

-----------------------------------------------------------------------------

![Entropy Increase with $\alpha$][multiAlpha_entropy_diffs]

-----------------------------------------------------------------------------

![Energy Increase with $\alpha$][multiAlpha_energy_diffs]

[multiAlpha_entropies]: img/20210728/entanglement_multiAlpha_entropies.pdf { .center height=98% }
[multiAlpha_energies]: img/20210728/entanglement_multiAlpha_energies.pdf { .center height=98% }
[multiAlpha_momentum]: img/20210728/entanglement_multiAlpha_momentum.pdf { .center height=98% }
[multiAlpha_entropy_diffs]: img/20210728/entanglement_multiAlpha_entropy_diffs.pdf { .center height=98% }
[multiAlpha_energy_diffs]: img/20210728/entanglement_multiAlpha_energy_diffs.pdf { .center height=98% }


# Varying $K$ and $\alpha$

----------------------------------------------------------------------------

![Entropy][multiK_multiAlpha_entropies]

-----------------------------------------------------------------------------

![Energy][multiK_multiAlpha_energies]

-----------------------------------------------------------------------------

![Momentum][multiK_multiAlpha_momentum]

-----------------------------------------------------------------------------

![Entropy Increase with K and $\alpha$][multiK_multiAlpha_entropy_diffs]

-----------------------------------------------------------------------------

![Energy Increase with K and $\alpha$][multiK_multiAlpha_energy_diffs]

[multiK_multiAlpha_entropies]: img/20210728/entanglement_multiK_multiAlpha_entropies.pdf { .center height=98% }
[multiK_multiAlpha_energies]: img/20210728/entanglement_multiK_multiAlpha_energies.pdf { .center height=98% }
[multiK_multiAlpha_momentum]: img/20210728/entanglement_multiK_multiAlpha_momentum.pdf { .center height=98% }
[multiK_multiAlpha_entropy_diffs]: img/20210728/entanglement_multiK_multiAlpha_entropy_diffs.pdf { .center height=98% }
[multiK_multiAlpha_energy_diffs]: img/20210728/entanglement_multiK_multiAlpha_energy_diffs.pdf { .center height=98% }

------------------------------------------------------------------------------

## Trial Math

- We use α from 0.1 to 0.8 and $α_c = 0.5$.

## References
