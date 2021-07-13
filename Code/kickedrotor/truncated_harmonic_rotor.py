import numpy as np
from scipy.integrate import solve_bvp
from scipy.special import hermite
import matplotlib.pyplot as plt
import functools

"""
We have two DE's:

psi'' = 2 (V(x) - E) psi
psi' = psi'

We take y = [psi, psi']
a = -pi, b = +pi

Boundary conditions:
psi(a) - psi(b) = 0
psi'(a) - psi'(b) = 0

"""

N = 100

def diffeqn(E, V, x, y):
    psi, dpsi = y
    coeff = 2 * (V(x) - E)
    dy = y[::-1]
    dy[1] *= coeff
    return dy

def potential(omega, alpha, x):
    condlist = [(x > alpha), np.logical_and((-alpha < x),(x < alpha)),  (x < -alpha)]
    harmonic = lambda t: 0.5 * omega**2 * (t**2 - alpha**2)
    return np.piecewise(x, condlist, [0, harmonic, 0])

def boundaries(ya, yb):
    return (ya - yb)

def solve(E, omega, alpha):
    V = functools.partial(potential, omega, alpha)
    f = functools.partial(diffeqn, E, V)

    x = np.linspace(-np.pi, np.pi, N)
    xi = np.sqrt(omega) * x
    psi_guess = (1/np.pi)**0.25 * hermite(0)(xi) * np.exp(-xi**2 / 2)
    dpsi_guess = np.append(np.diff(psi_guess), [0])
    y = np.array([psi_guess, dpsi_guess])
    params = {
        "fun": f,
        "bc": boundaries,
        "x": x,
        "y": y,
        "max_nodes": 1000000
    }

    sol = solve_bvp(**params)
    if sol.success:
        return sol
    else:
        print(f"Integration failed: {sol.message}")
        return None

def plotter(sol):
    x = sol.x
    psi, dpsi = sol.y
    intp_psi, intp_dpsi = sol.sol

    plt.plot(x, psi, label=r"$\psi$")
    plt.plot(x, dpsi, label=r"$\psi'$")
    plt.plot(x, intp_psi(x), label=r"$\psi_{int}$")
    plt.plot(x, intp_dpsi(x), label=r"$\psi_{int}'$")
    plt.legend()


if __name__ == "__main__":
    sol = solve(-45, 10, 1)
    if sol is not None:
        plotter(sol)
        plt.show()
