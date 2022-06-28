import matplotlib.pyplot as plt
import numpy as np


def plot_result(x: np.ndarray, u: np.ndarray, t: np.ndarray):
    (n_steps, n_states) = x.shape
    x_angles = x[:, :int(n_states / 2)]
    x_dangles = x[:, int(n_states / 2):]

    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True)
    axs[0].plot(t, x_angles)
    axs[1].plot(t, x_dangles)
    axs[2].plot(t[:-1], u)

    axs[0].set_ylabel(r"$q$ ($rad$)")
    axs[1].set_ylabel(r"$\dot{q}$ ($\frac{rad}{s}$)")
    axs[2].set_ylabel(r"$\tau$ ($Nm$)")

    axs[2].set_xlabel(r"$t$ ($s$)")
    [ax.grid() for ax in axs]
    plt.show()


def print_timings(t_mean: float, t_std: float, t_min: float, t_max: float):
    print("Solver times: ")
    print("\tmean: \t{:10.4f}s, \n"
          "\tstd: \t{:10.4f}s, \n"
          "\tmin: \t{:10.4f}s, \n"
          "\tmax:\t{:10.4f}s".format(t_mean, t_std, t_min, t_max))
