import numpy as np
from scipy.fft import fft, fftfreq, fftshift, fftn, ifftn
from scipy.special import jv, seterr, errstate
import matplotlib.pyplot as plt

# Program Constants
N = 101
K = 5
ALPHA = 0.1

def f(t):
    return np.exp(-1j * K * np.cos(t))

def f3d(theta1, theta2, theta3):
    quasi = (1 + ALPHA * np.cos(theta2) * np.cos(theta3))
    return np.exp(-1j * K * np.cos(theta1) * quasi)

def expected(freq):
    return jv(-freq, -K) * (1j)**(-freq)

def main1d():
    t = 2 * np.pi * np.arange(N) / N
    x = f(t)

    y = fftshift(fft(x, norm="forward"))
    freq = fftshift(fftfreq(N, d=2*np.pi/N))
    int_freq = np.arange(-N/2, N/2, dtype=int)

    y_exp = expected(int_freq)
    # fail_freq = int_freq[np.isnan(y_exp)]
    # print("Failing Frequencies:", fail_freq)
    # with errstate(all="raise"):
        # print("-K:", -K)
        # print(jv(-fail_freq, -K))


    # print(geterr())
    diff = np.abs(y_exp - y)
    # print(y_exp)
    print(f"Max diff: {diff.max()}")

    plt.plot(freq, diff.real, label="Real")
    plt.plot(freq, diff.imag, label="Imag")
    plt.legend()
    plt.show()

def main3d():
    theta = 2 * np.pi * np.arange(N) / N
    T1, T2, T3 = np.meshgrid(theta, theta, theta)
    X = f3d(T1, T2, T3)

    Y = fftn(X, norm="forward")
    # freq = fftshift(fftfreq(N)) * N
    # F1, F2, F3 = np.meshgrid(freq, freq, freq)

    X_back = ifftn(Y, norm="forward")
    print("Max diff:", np.abs(X - X_back).max())

if __name__ == "__main__":
    # seterr(all="raise")
    main3d()
