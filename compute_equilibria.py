from copy import copy

import matplotlib.pyplot as plt
import numpy as np

from flexible_arm import SymbolicFlexibleArm, FlexibleArm
import casadi as cs
from controller import ConstantController
from simulation import simulate_closed_loop


def compute_equilibria(model: SymbolicFlexibleArm, torque_max: float = 8.401) -> [np.ndarray, np.ndarray]:
    # definition of variables and parameters
    n_disc = 100
    u_equi = np.linspace(0, torque_max, n_disc)
    x_equi = np.zeros((2, model.nx, n_disc))
    u_equi_out = np.vstack((u_equi, u_equi[::-1]))

    x = cs.MX.sym("x", int(model.nx))
    for i_starting_point, initial_state in enumerate([np.pi / 2, -np.pi / 2]):
        ini_vec = [0.0] * 10
        ini_vec[0] = initial_state
        for i, u in enumerate(u_equi):
            res = model.ode(x, u)
            eq_fun = cs.Function("equi_fun", [x], [res])
            rf = cs.rootfinder('rf', 'kinsol', eq_fun)
            x_equi[i_starting_point, :, i] = rf(ini_vec).full()[:, 0]
            ini_vec = copy(x_equi[i_starting_point, :, i])
    return x_equi, u_equi_out


def forward_kinematik(x: np.ndarray, model: SymbolicFlexibleArm):
    (n_states, n_equis) = x.shape
    assert n_states == model.nx
    n_segments = int(model.nx / 2)
    len_segment = model.length / n_segments
    pos_x = np.zeros((n_equis, 1))
    pos_y = np.zeros((n_equis, 1))
    for i_equi in range(n_equis):
        cum_angle = 0
        for seg in range(n_segments):
            cum_angle += x[seg, i_equi]
            pos_x[i_equi, 0] += len_segment * np.cos(cum_angle)
            pos_y[i_equi, 0] += len_segment * np.sin(cum_angle)

    return pos_x, pos_y


def plot_equilibria(x_equi: np.ndarray, model: SymbolicFlexibleArm):
    for sign, label_sign in zip([-1, 1], ["negative", "positive"]):
        for i_equilibirum, label_equi in zip([0, 1], ["upper", "lower"]):
            x_equi_transform = copy(x_equi[i_equilibirum, :, :])
            x_equi_transform[0] += np.pi / 2
            x_equi_transform = x_equi_transform * sign
            x_equi_transform[0] -= np.pi / 2
            pos_x, pos_y = forward_kinematik(x=x_equi_transform, model=model)
            plt.plot(pos_x, pos_y, label=label_sign + ", " + label_equi + " equilibrium")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.grid()
    plt.legend()
    plt.title("Equilibria of flexible arm")
    plt.axis("equal")
    plt.show()



if __name__ == "__main__":
    fa_sym = SymbolicFlexibleArm(K=[7] * 4)
    fa = FlexibleArm(K=[7.] * 4)
    x_equi, u_equi = compute_equilibria(model=fa_sym)
    plot_equilibria(x_equi=x_equi, model=fa_sym)
