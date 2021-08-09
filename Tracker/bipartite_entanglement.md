---
title: Bipartite Entanglement Entropy
author: Aditya Chincholi
date: June 28, 2021
theme: metropolis
aspectratio: 169
bibliography: citations.bib
...

# Quasiperiodic Kicked Rotor

## What I did earlier

- Constructing the full floquet operator in a single basis.
- Using a density matrix the whole time for calculations.
- This has the drawback of increasing computational complexity
    of each individual step and the memory used at any given
    time is large.

## What I have done now
- Separate the floquet operator into momentum space and position
    space parts.
- Use a single pure state for calculations.
- Fourier transform the state at each time step and apply both parts
    of the floquet operator in their resp. basis.
- This is better as the memory used is less but computation increases.
    Since fourier transforms are computationally cheap anyway, so it's
    fine.
- Peak memory required scales the same way but we have reduced
    it by a constant factor and it is not used in all calculations.

## Results

We use $\hbar = 2.85, \omega_2 = 2\pi\sqrt{5},
\omega_3 = 2\pi\sqrt{13}$, the momentum ranges from -10 to 10

$$
H = \frac{p_1^2}{2} + p_2 \omega_2 + p_3 \omega_3 + K cos(\theta_1)
(1 + \alpha cos(\theta_2) cos(\theta_3)) \sum_n \delta (t - n)
$$

--------------------------------------------------------------

![Precritical (Insulator): $K = 4, \alpha = 0.2$](./img/quasiperiodic_entropies_insulator.pdf){height=90%}

--------------------------------------------------------------

![Critical: $K = 6.36, \alpha = 0.4375$](./img/quasiperiodic_entropies_critical.pdf){height=90%}

--------------------------------------------------------------

![Post-critical (Metal): $K = 8, \alpha = 0.8$](./img/quasiperiodic_entropies_metallic.pdf){height=90%}

--------------------------------------------------------------


- I don't see much of a trend here. The entanglement grows faster
    and higher with higher K values i.e. more diffusive the regime
    higher the entanglement for the same number of time steps but
    other than that, I don't see anything here.

-------------------------------------------------------------

## Bypassing The Density Matrix

- What I was doing: $\ket{\psi} \rightarrow \rho \rightarrow \rho_1 = Tr_{2,3}[\rho]$.
    This is a memory bottleneck as the full density matrix $\rho$ still
    requires the $O(N^6)$ memory.

- What I realized: $\ket{\psi} \rightarrow \rho_1 = Tr_{2,3}[\rho]$
    This gives a really high speedup and allows me to check if localisation
    is happening properly or not.

- I used these params to generate the following results:
    p-basis = [-100, 100], 80 timesteps, $\omega_2 = 2\pi\sqrt{5}$,
    $\omega_3 = 2\pi\sqrt{13}$, $\hbar$ = 2.85.

- We plot the following quantities:
    * $P(p_1 = m \hbar)$
    * $E = p_1^2/2 + p_2 \omega_2 + p_3 \omega_3$
    * $S = -\rho_1 ln(\rho_1)$

--------------------------------------------------------------

## Momentum ($p_1$) distributions

![K = 3, $\alpha$ = 0.1](img/quasiperiodic_momenta_N100_ALPHA0.100.pdf){ .center height=70% }

--------------------------------------------------------------

## Momentum ($p_1$) distributions

![K = 6.36, $\alpha$ = 0.4375](img/quasiperiodic_momenta_N100_ALPHA0.438.pdf){ .center height=70% }

--------------------------------------------------------------

## Momentum ($p_1$) distributions

![K = 7, $\alpha$ = 0.8](img/quasiperiodic_momenta_N100_ALPHA0.800.pdf){ .center height=70% }

--------------------------------------------------------------

## Energy

![K = 3, $\alpha$ = 0.1](img/quasiperiodic_energies_N100_ALPHA0.100.pdf){ .center height=70% }

--------------------------------------------------------------

## Energy

![K = 6.36, $\alpha$ = 0.4375](img/quasiperiodic_energies_N100_ALPHA0.438.pdf){ .center height=70% }

--------------------------------------------------------------

## Energy

![K = 7, $\alpha$ = 0.8](img/quasiperiodic_energies_N100_ALPHA0.800.pdf){ .center height=70% }

--------------------------------------------------------------

## Entropy

![K = 3, $\alpha$ = 0.1](img/quasiperiodic_entropies_N100_ALPHA0.100.pdf){ .center height=70% }

--------------------------------------------------------------

## Entropy

![K = 6.36, $\alpha$ = 0.4375](img/quasiperiodic_entropies_N100_ALPHA0.438.pdf){ .center height=70% }

--------------------------------------------------------------

## Entropy

![K = 7, $\alpha$ = 0.8](img/quasiperiodic_entropies_N100_ALPHA0.800.pdf){ .center height=70% }

---------------------------------------------------------------

## Multiple K Values

::: nonincremental

- We look at the following plots for the following at different $K, \alpha$
    values from the insulator to the metallic regime:
    1. Energy Expectation Value
    2. Entanglement Entropy
    3. Momentum Distribution

- To study the changes in the energy and entropy values with K, we have
    plotted the following quantities:
    1. Entropy Difference: $S(K_{n+1}, \alpha_{n+1}) - S(K_n, \alpha_n)$ vs t
    2. Energy Difference: $E(K_{n+1}, \alpha_{n+1}) - E(K_n, \alpha_n)$ vs t

:::
---------------------------------------------------------------------------

## Multiple K Values

::: nonincremental

- First we take a big picture look with 11 values with $K \in [3.00, 9.72]$
    and $\alpha \in [0.200, 0.6750]$. Both ranges are centred around the
    critical point[^1]. Momentum range is -100 to 100 with 80 timesteps.

- Then we look very close to the critical point with 11 values with
    $K \in [6.30, 6.42]$ and $\alpha \in [0.4000, 0.4750]$. Both ranges
    are centred around the critical point. Momentum range is -100 to 100
    with 80 timesteps.



[^1]: The [@lemarieUniversalityAndersonTransition2009] paper gives the value
    of K at critical point, but not of $\alpha$.

:::

----------------------------------------------------------------------------

The Big Picture

----------------------------------------------------------------------------

![](img/20210721/quasiperiodic_energies_multik_macro.pdf){ .center }

----------------------------------------------------------------------------

![](img/20210721/quasiperiodic_entropies_multik_macro.pdf){ .center }

----------------------------------------------------------------------------

![](img/20210721/quasiperiodic_momentum_multik_macro.pdf){ .center }

----------------------------------------------------------------------------

![](img/20210721/quasiperiodic_energy_diffs_multik_macro.pdf){ .center }

----------------------------------------------------------------------------

![](img/20210721/quasiperiodic_entropy_diffs_multik_macro.pdf){ .center }

----------------------------------------------------------------------------

The Microscopic Picture

----------------------------------------------------------------------------

![](img/20210721/quasiperiodic_energies_multik_micro.pdf){ .center }

----------------------------------------------------------------------------

![](img/20210721/quasiperiodic_entropies_multik_micro.pdf){ .center }

----------------------------------------------------------------------------

![](img/20210721/quasiperiodic_momentum_multik_micro.pdf){ .center }

----------------------------------------------------------------------------

![](img/20210721/quasiperiodic_energy_diffs_multik_micro.pdf){ .center }

----------------------------------------------------------------------------

![](img/20210721/quasiperiodic_entropy_diffs_multik_micro.pdf){ .center }

----------------------------------------------------------------------------

##

----------------------------------------------------------------------------

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

## References
