import warnings
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate

from flexible_arm import SymbolicFlexibleArm, FlexibleArm
import casadi as cs
from warnings import warn


def compute_equilibria(model: SymbolicFlexibleArm, torque_max: float) -> [np.ndarray, np.ndarray]:
    # K=7 => torque_max=8.401
    # definition of variables and parameters
    n_disc = 100
    u_equi = np.linspace(-torque_max, torque_max, n_disc)
    x_equi = np.zeros((2, model.nx, n_disc))
    u_equi_out = np.vstack((u_equi, u_equi))

    x = cs.MX.sym("x", int(model.nx))
    for i_starting_point, initial_state in enumerate([np.pi / 2, -np.pi / 2]):
        ini_vec = [0.0] * model.nx
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
    return_val = np.zeros((n_equis, 2))
    return_val[:, 0] = pos_x[:,0]
    return_val[:, 1] = pos_y[:,0]
    return return_val


def plot_equilibria(x_equi: np.ndarray, model: SymbolicFlexibleArm):
    for sign, label_sign in zip([-1, 1], ["negative", "positive"]):
        for i_equilibirum, label_equi in zip([0, 1], ["upper", "lower"]):
            x_equi_transform = copy(x_equi[i_equilibirum, :, :])
            x_equi_transform[0] += np.pi / 2
            x_equi_transform = x_equi_transform * sign
            x_equi_transform[0] -= np.pi / 2
            pos = forward_kinematik(x=x_equi_transform, model=model)
            pos_x = pos[:,0]
            pos_y = pos[:,1]
            plt.plot(pos_x, pos_y, label=label_sign + ", " + label_equi + " equilibrium")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.grid()
    plt.legend()
    plt.title("Equilibria of flexible arm")
    plt.axis("equal")
    plt.show()


class EquilibriaWrapper:
    def __init__(self, model_sym: SymbolicFlexibleArm, model: FlexibleArm, guess_max_torque: float = 0.5):
        self.model_symbolic = model_sym
        self.model = model
        self.torque_max = guess_max_torque
        x_equi, u_equi = compute_equilibria(model=model_sym, torque_max=guess_max_torque)
        self.x_equi = x_equi
        self.n_equi = x_equi.shape[0]
        self.interpolators = []
        for i in range(self.n_equi):
            self.interpolators.append(scipy.interpolate.interp1d(u_equi[i, :], x_equi[i, :]))

    def get_equilibrium_kin_states(self, input_u: float):
        if input_u > self.torque_max:
            warn("Torque outside (computed) equilibrium range!")
            return None
        return np.array([fun(input_u) for fun in self.interpolators])

    def get_equilibrium_cartesian_states(self, input_u: float):
        x_equis = self.get_equilibrium_kin_states(input_u=input_u)
        x_cartesian_equis = forward_kinematik(x=np.moveaxis(x_equis, 0, 1), model=self.model_symbolic)
        return x_cartesian_equis

    def plot(self):
        plot_equilibria(x_equi=self.x_equi, model=self.model_symbolic)


if __name__ == "__main__":
    n_links = 10
    fa_sym = SymbolicFlexibleArm(n_links)
    fa = FlexibleArm(n_links)
    equi_wrapper = EquilibriaWrapper(model_sym=fa_sym, model=fa)
    print(equi_wrapper.get_equilibrium_cartesian_states(input_u=0))
    equi_wrapper.plot()
