#!/usr/bin/env python3

import os
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
from animation import Panda3dAnimator


class Poly5Trajectory:
    """
    NOTE The class is primarily used for velocity reference generation
    for point-to-point motion therefore some properties and attributes
    related to position reference generation might not be implemented.
    It is especially true for rotation motion!!!!
    """

    def __init__(self, yi, yf, tf, ts=0.001) -> None:
        """
        :param yi: initial position, can be joint position or cartesian position
        :type yi: a numpy matrix
        :param yf: terminal position, -/-
        :type yf: a numpy matrix
        :param tf: travel time
        :param ts: sampling time
        """
        self.pi = yi
        self.pf = yf
        self.ts = ts
        self.tf = tf
        self.t = np.arange(0, tf + ts, ts).reshape(-1, 1)

        # get interpolation function and desing trajectory
        self.r, self.dr, self.ddr = self.interp_fcn(self.t, self.tf)
        self.p, self.dp, self.ddp = self.design_traj()

    @staticmethod
    def interp_fcn(t, tf):
        """ Interpolation function for quintic polynomial. For 
        more information refer to "Modeling, Identification and Control of Robots"

        :parameter t: [Nx1] time samples
        :parameter tf: travel time
        """
        ttf = t / tf
        r = 10 * ttf ** 3 - 15 * ttf ** 4 + 6 * ttf ** 6
        dr = 1 / tf * (30 * ttf ** 2 - 60 * ttf ** 3 + 30 * ttf ** 4)
        ddr = 1 / tf ** 2 * (60 * ttf - 180 * ttf ** 2 + 120 * ttf ** 3)
        return r, dr, ddr

    def design_traj(self):
        """ Design a trajectory 
        """
        # Translational motion
        delta_p = self.pf - self.pi  # amplitude
        p = self.pi.T + self.r * delta_p.T
        dp = self.dr * delta_p.T
        ddp = self.ddr * delta_p.T

        return p, dp, ddp

    @property
    def position_reference(self):
        return self.p

    @property
    def velocity_reference(self):
        return self.dp

    @property
    def acceleration_reference(self):
        return self.ddp


def get_reference_for_all_joints(q_t0, pee_tf, tf, ts, n_seg):
    # Only get active joints
    q_t0_active = np.array([q_t0[0], q_t0[1], q_t0[1 + (n_seg + 1)]])

    # compute initial guesses for active joints
    t, q, dq = initial_guess_for_active_joints(q_t0_active, pee_tf, tf, ts)

    # Expand to full states again
    n_t, _ = q.shape
    q_full = np.zeros((n_t, 1 + 2 * (n_seg + 1)))
    dq_full = np.zeros((n_t, 1 + 2 * (n_seg + 1)))
    q_full[:, 0] = q[:, 0]
    q_full[:, 1] = q[:, 1]
    q_full[:, 1 + n_seg + 1] = q[:, 2]
    dq_full[:, 0] = dq[:, 0]
    dq_full[:, 1] = dq[:, 1]
    dq_full[:, 1 + n_seg + 1] = dq[:, 2]

    return t, q_full, dq_full


def initial_guess_for_active_joints(q_t0, pee_tf, tf, ts):
    """ Computes an initial guess for active joints of the
    3dof arm using quintic polynomial trajectory

    :parameter q_t0: positions of the active joints at t0
    :parameter pee_tf: position of the end-effector at tf
    :parameter tf: travel time
    :parameter ts: sampling time
    """
    # Load functions needed for computing forward kinematics
    eval_pee = cs.Function.load('models/three_dof/zero_segments/fkp.casadi')

    # Compute the end-effector position at t0
    pee_t0 = np.array(eval_pee(q_t0))

    # Design polynomial trajectory
    poly5 = Poly5Trajectory(pee_t0, pee_tf, tf, ts)
    v = poly5.velocity_reference
    a = poly5.acceleration_reference
    t = np.arange(0, tf + ts, ts)

    # Load a jacobian function for evaluating jacobian
    eval_Jee = cs.Function.load('models/three_dof/zero_segments/fkJ.casadi')

    # Perform inverse kinematics at velocity level
    ns = v.shape[0]
    dq = np.zeros((ns, 3))
    q = np.zeros((ns, 3))
    q[0, :] = q_t0.flatten()
    for k, (qk, vk, ak) in enumerate(zip(q, v, a)):
        Jk = np.array(eval_Jee(qk))
        dq[k, :] = np.linalg.solve(Jk, vk)

        if k < ns - 1:
            q[k + 1, :] = q[k, :] + ts * dq[k, :]

    return t, q, dq


def plot_trajectory(t, y, labels):
    _, ax = plt.subplots()
    for dp, l in zip(y.T, labels):
        ax.plot(t, dp, label=l)
    ax.set_xlabel("t (sec)")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load functions needed for computing forward kinematics
    eval_pee = cs.Function.load('models/three_dof/zero_segments/fkp.casadi')

    # Specify a task
    q_t0 = np.array([[0, np.pi / 8, np.pi / 6]]).T
    pee_t0 = np.array(eval_pee(q_t0))

    delta_pee = np.array([[0, 0, -0.2]]).T
    pee_tf = pee_t0 + delta_pee

    tf = 0.5
    ts = 0.01

    # Design an initial guess
    t, q, dq = initial_guess_for_active_joints(q_t0, pee_tf, tf, ts)

    # Plot or animate an initial guess
    # plot_trajectory(t, q, ['q1', 'q2', 'q3'])

    urdf_path = 'models/three_dof/zero_segments/flexible_arm_3dof_0s.urdf'
    animator = Panda3dAnimator(urdf_path, ts, q).play(3)
