# Classical Kicked Rotor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

k = 8 #2.7075 seems to be around the turning point
n = 10000

def step(θ_p, L_p):
    θ_n = (θ_p + L_p) % (2*np.pi)
    L_n = L_p + k*np.sin(θ_p)
    return θ_n, L_n

def run(θ_0, L_0):
    θs = [θ_0]
    Ls = [L_0]

    for t in range(n):
        θ, L = step(θs[-1], Ls[-1])
        θs.append(θ)
        Ls.append(L)

    return np.array(θs), np.array(Ls)

def getInitialConditions():
    θvals = np.linspace(0, 2*np.pi, 100)
    Lvals = 1 + np.zeros(θvals.shape)
    return θvals, Lvals

def plotLdistribution(θs, Ls):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(6,10))
    for t in range(0, n, 1000):
        sns.kdeplot(Ls[t, :].flatten(), ax=ax1, label=str(t))

    ax2.plot(np.arange(n+1), np.var(Ls, axis=1))
    ax1.legend()
    ax1.set_xlabel("L")
    ax1.set_ylabel("Density")
    ax2.set_xlabel("t")
    ax2.set_ylabel("var(L)")
    fig.suptitle(f"k = {k}")
    plt.tight_layout()
    plt.savefig("plots/classical_kicked_rotor.png")

if __name__ == "__main__":
    θvals, Lvals = getInitialConditions()
    θs, Ls = run(θvals, Lvals)
    plotLdistribution(θs, Ls)
    plt.show()
