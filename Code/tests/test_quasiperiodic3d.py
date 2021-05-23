"""
Author: Aditya Chincholi
Purpose: To test that the floquet operator for the 3d
        quasiperiodic kicked rotor is working correctly.
"""

import numpy as np
from scipy.integrate import nquad
import kickedrotor.quasiperiodic_rotor_3d as rotor
from kickedrotor.params import *

integral_cache = {}

def integrand_real(t1, t2, t3, r1, r2, r3):
    kick = -(K/HBAR) * np.cos(t1) * (1 + ALPHA * np.cos(t2) * np.cos(t3))
    kernel = -r1*t1 - r2*t2 - r3*t3
    return np.cos(kick + kernel)

def integrand_imag(t1, t2, t3, r1, r2, r3):
    kick = -(K/HBAR) * np.cos(t1) * (1 + ALPHA * np.cos(t2) * np.cos(t3))
    kernel = -r1*t1 - r2*t2 - r3*t3
    return np.sin(kick + kernel)

def computeElement(m, n):
    m1, m2, m3 = m
    n1, n2, n3 = n
    r1, r2, r3 = (m1-n1, m2-n2, m3-n3)

    global integral_cache

    if (r1, r2, r3) in integral_cache.keys():
        integral = integral_cache[(r1, r2, r3)]

    else:
        ranges = [(0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi)]
        integral_real, err_real = nquad(integrand_real, ranges, args=(r1, r2, r3))
        integral_imag, err_imag = nquad(integrand_imag, ranges, args=(r1, r2, r3))
        integral = (integral_real + 1j * integral_imag) / (2 * np.pi)**3
        integral_cache[(r1, r2, r3)] = integral

    result = integral * np.exp(-1j * (HBAR * n1**2 + n2 * OMEGA2 + n3 * OMEGA3))
    return result

def changeBase(m, base):
    m1 = m // base**2
    remainder = m % base**2
    m2 = remainder // base
    remainder = remainder % base
    m3 = remainder
    return m1, m2, m3

def computeDiagonal():
    base_case = computeElement((0, 0, 0), (0, 0, 0))
    print(f"Base case integral: {base_case}")
    col = np.arange(0, DIM**3)
    n1, n2, n3 = changeBase(col, DIM)
    n1 -= N
    n2 -= N
    n3 -= N
    return np.exp(-1j * ((HBAR/2) * n1**2 + n2 * OMEGA2 + n3 * OMEGA3)) * base_case
    # return base_case

def computeFirstDiagonals():
    diagonal_lower = np.empty(DIM**3 - 1, dtype=DTYPE)

    for i in range(DIM**3 - 1):
        row = i + 1
        col = i
        m1, m2, m3 = changeBase(row, DIM)
        n1, n2, n3 = changeBase(col, DIM)
        m1 -= N # I know this is ugly
        m2 -= N
        m3 -= N
        n1 -= N
        n2 -= N
        n3 -= N
        diagonal_lower[i] = computeElement((m1, m2, m3), (n1, n2, n3))

    diagonal_upper = np.empty(DIM**3 - 1, dtype=DTYPE)

    for i in range(DIM**3 - 1):
        row = i
        col = i + 1
        m1, m2, m3 = changeBase(row, DIM)
        n1, n2, n3 = changeBase(col, DIM)
        m1 -= N # I know this is ugly
        m2 -= N
        m3 -= N
        n1 -= N
        n2 -= N
        n3 -= N
        diagonal_upper[i] = computeElement((m1, m2, m3), (n1, n2, n3))

    return diagonal_upper, diagonal_lower

def mainDiagonalTest(observed_diagonal):
    expected_diagonal = computeDiagonal()
    np.testing.assert_allclose(expected_diagonal, observed_diagonal)

def firstDiagonalTest(observed_diagonal_upper, observed_diagonal_lower):
    expected_diagonal_upper, expected_diagonal_lower = computeFirstDiagonals()

    diff_upper = np.abs(expected_diagonal_upper - observed_diagonal_upper)
    diff_lower = np.abs(expected_diagonal_lower - observed_diagonal_lower)

    diff_index_upper = np.where(diff_upper > 1e-9)
    diff_index_lower = np.where(diff_upper > 1e-9)

    print(f"Upper Difference at: {diff_index_upper}")
    print(f"Lower Difference at: {diff_index_lower}")

    print("Observed Upper:", observed_diagonal_upper[diff_index_upper])
    print("Expected Upper:", expected_diagonal_upper[diff_index_upper])
    print("Observed Lower:", observed_diagonal_lower[diff_index_lower])
    print("Expected Lower:", expected_diagonal_lower[diff_index_lower])

    np.testing.assert_allclose(expected_diagonal_lower, observed_diagonal_lower, atol=1e-10)
    np.testing.assert_allclose(expected_diagonal_upper, observed_diagonal_upper, atol=1e-10)

def test_main():
    """
    Runs all the tests on a floquet matrix from the code.
    We run all the tests from here as it is expensive to
    compute the floquet matrix.
    """
    F = rotor.getFloquetOperator()
    mainDiagonalTest(np.diag(F))
    firstDiagonalTest(np.diag(F, k=1), np.diag(F, k=-1))
