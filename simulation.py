#!/usr/bin/env python3

import numpy as np
import pinocchio as pin
from scipy.integrate import solve_ivp

from flexible_arm import FlexibleArm
from animation import Animator
from controller import DummyController, PDController


class Simulator:
    """ Implements a simulator for FlexibleArm
    """

    def __init__(self) -> None:
        pass

    def __ode(self, t, x):
        """ Wraps ode of the FlexibleArm to match scipy notation
        """


def robot_ode(t, x, robot, tau):
    """ Wrapper on ode of the robot class
    """
    return robot.ode(x, tau)


def simulate_closed_loop(ts, n_iter, robot, controller, x0):
    x = np.zeros((n_iter + 1, robot.nx))
    u = np.zeros((n_iter, robot.nu))
    x[0, :] = x0
    for k in range(n_iter):
        qk = x[[k], :robot.nq].T
        dqk = x[[k], robot.nq:].T

        tau = controller.compute_torques(qk, dqk)
        u[[k], :] = tau

        sol = solve_ivp(robot_ode, [0, ts], x[k, :], args=(robot, tau), vectorized=True)
        x[k + 1, :] = sol.y[:, -1]

    return x, u


if __name__ == "__main__":
    # Create FlexibleArm instance
    fa = FlexibleArm()

    # Sample a random configuration
    q = pin.randomConfiguration(fa.model)
    # fa.visualize(q)

    # Simulate
    q = np.zeros((fa.nq, 1))
    dq = np.zeros_like(q)
    x0 = np.vstack((q, dq))

    # controller = DummyController()
    controller = PDController(Kp=150, Kd=5, q_ref=np.array([np.pi / 2]))

    ts = 0.01
    n_iter = 150
    x, u = simulate_closed_loop(ts, n_iter, fa, controller, x0.flatten())

    # Parse joint positions
    q = x[:, :fa.nq]

    # Animate simulated motion
    anim = Animator(fa, q).animate()
