from enum import Enum, auto
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

n_seg_int2str = {0: "zero", 1: "one", 2: "two", 3: "three", 5: "five", 10: "ten"}


class ControlMode(Enum):
    SET_POINT = auto()
    REFERENCE_TRACKING = auto()


class StateType(Enum):
    REAL = auto()
    ESTIMATED = auto()


def plot_result(x: np.ndarray, u: np.ndarray, t: np.ndarray):
    (n_steps, n_states) = x.shape
    x_angles = x[:, : int(n_states / 2)]
    x_dangles = x[:, int(n_states / 2) :]

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


def print_timings(
    t_mean: float, t_std: float, t_min: float, t_max: float, name="solver"
):
    print(name + " times: ")
    print(
        "\tmean: \t{:10.4f}s, \n"
        "\tstd: \t{:10.4f}s, \n"
        "\tmin: \t{:10.4f}s, \n"
        "\tmax:\t{:10.4f}s".format(t_mean, t_std, t_min, t_max)
    )


def seed_everything(seed: int) -> None:
    """
    Taken and modified from Lightning: https://github.com/Lightning-AI/lightning/blob/725159ed604673e847b7821b08dceff80ec9c735/src/lightning/fabric/utilities/seed.py
    Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
