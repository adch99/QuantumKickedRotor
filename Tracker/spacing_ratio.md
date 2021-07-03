---
title: Spacing Ratio Distribution
author: Aditya Chincholi
date: 28 June, 2021
theme: metropolis
aspectratio: 169
bibliography: citations.bib
...

## First: A Minor Problem

- Is time reversal symmetry $t \to -t$?

- Or is it $t \to -t$, $\theta \to -\theta$, $p \to p$?

- The issue is the second definition is the one used by Lemarie et al
    in the "Universality of the Anderson Transition" paper
    [@lemarieUniversalityAndersonTransition2009].

-------------------------------------------------------------------

## The Problem
$$ H = \frac{p^2}{2} + K cos(\theta)\sum_n \delta(t - n\tau) $$

This hamiltonian is symmetric wrt $t \to -t$, $\theta \to -\theta$
i.e time reversal and wrt $\theta \to -\theta$, $p \to -p$ i.e. parity.
The time reversal only holds if we consider only $\Delta t = N \tau$
but that is fine I guess.

---------------------------------------------------------------------

## Parity is a headache

- To remind you, I got very odd graphs for the standard kicked rotor
    system wherein the spacing and ratio distributions were primarily
    concentrated near zero even after mean normalization (i.e. unfolding).
- So I tried to study up Mehta's book better and try to understand the issue.
- Since the Hamiltonian has parity symmetry, I think that is what is causing
    the problem. The eigenvectors have a parity quantum number which needs
    to be separated.
- ~~But even then, the distribution should be $e^{-s}$ for the spacing. which
    is not the case.~~
- The Izrailev and Atas papers I was talking about last time were indeed on
    kicked rotors on a torus as you suspected.

---------------------------------------------------------------------

## Parity is a headache (cont.)

- So I tried separating out the eigenvectors by determining their parity,
    but the eigenvectors I obtained were not of sufficient accuracy to
    distinguish the parity. They are good for a 100x100 matrix, but for
    a 1000x1000 matrix, the eigenvectors are not accurate enough for me
    to do this.
- I thought that maybe the eigenvectors I was getting were linear combos
    so I tried to understand how to simultaneously diagonalize two matrices.
- Apparently, we just take an arbitrary linear combination of the two
    matrices and solve for the eigenvectors of this linear combination to
    get the required eigenvectors. I tried it, but it wasn't very successful.
- Maybe I'm missing something but apparently, eigenvector solving doesn't
    allow me to specify accuracy/tolerance anywhere.
- ~~All in all, I have no clue what is going wrong.~~

---------------------------------------------------------------------------

## Finding the root

- In order to find out the main issue with my code, I began to try and
    generate random matrices from different classes as test cases.
- As it turns out, people write a great deal of theoretical analysis
    but rarely write about how to generate them numerically. The
    symplectic ensembles are particularly hard to find in this case.
- I found a few places where it was detailed and I managed to implement
    GUE, GOE, GSE, CUE, COE, CSE generators. [@mezzadriHowGenerateRandom2007;
    @edelmanRandomMatrixTheory2005; @sircaComputationalMethodsPhysicists2012]
- It turned out that my program worked correctly for the unitary and
    orthogonal ensembles but failed for the symplectic ensembles
    because they show degeneracy.

---------------------------------------------------------------------

## Finding the root (cont.)

- The symplectic ensembles show Kramer's degeneracy due to the half
    integral spin. Because of that, each energy level has at least
    a double degeneracy (up and down spin for spin $\frac{1}{2}$).
- So by writing code that kept only unique eigenvalues (or eigenphases),
    the issue got resolved.
- And this worked for the earlier case of the quantum kicked rotor
    as well. Using the parity eigenvalue would be more accurate in
    some sense because we don't remove other symmetries that may
    exist that we don't know of, but it is not good numerically.
- This method removes the degeneracies arising from all the
    symmetries, but if we are sure that there is only one symmetry
    this will work. By taking the ratio of number of unique
    eigenvalues to the original number of eigenvalues, we can find
    the mean degeneracy.

-------------------------------------------------------------------------

## Results

![Standard kicked rotor with $K = 5$, $p$-basis $[-2000, 2000]$, 3991
eigenphases used and 10 discarded (tol = $0.1$). Uniqueness ratio is
0.52 with tol = $10^{-8}$](img/quantum_kicked_rotor_N2000_K5_spacing_ratios.pdf){ height=70% .center }

-------------------------------------------------------------------------
## References
