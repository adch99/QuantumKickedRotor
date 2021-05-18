import numpy as np
import matplotlib.pyplot as plt
import kickedrotor.quasiperiodic_kicked_rotor as rotor


def main():
    F = rotor.floquetOperator(5, 5, alpha=0.8)
    eigs = np.sort(np.abs(np.linalg.eigvals(F)))

    plt.hist(eigs)
    plt.yscale("log")
    plt.xlabel("Magnitude of Eigenvalue")
    plt.ylabel("Number of Values")
    plt.title("Distribution of Eigenvalues for Quasiperiodic Floquet Operator")
    plt.savefig("plots/quasiperiodic_floquet_eigenval_dist_10.png")
    plt.show()

if __name__ == "__main__":
    main()
