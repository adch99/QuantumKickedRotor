---
title: Bipartite Entanglement Entropy
author: Aditya Chincholi
date: June 28, 2021
aspectratio: 169
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
