#!/usr/bin/env python3

import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from flexible_arm import FlexibleArm
from animation import Animator
from controller import DummyController, PDController


class Simulator:
    """ Implements a simulator for FlexibleArm
    """
    def __init__(self, robot, controller, integrator) -> None:
        self.robot = robot
        self.controller = controller
        self.integrator = integrator

    @staticmethod
    def ode_wrapper(t, x, robot, tau):
        """ Wraps ode of the FlexibleArm to match scipy notation
        """
        return robot.ode(x, tau)

    def simulate(self, x0, ts, n_iter):
        x = np.zeros((n_iter+1, self.robot.nx))
        u = np.zeros((n_iter, self.robot.nu))
        x[0,:] = x0
        for k in range(n_iter):
            qk = x[[k],:self.robot.nq].T
            dqk = x[[k],self.robot.nq:].T

            tau = controller.compute_torques(qk[0], dqk[0])
            u[[k],:] = tau

            sol = solve_ivp(self.ode_wrapper, [0, ts], x[k,:], args=(self.robot, tau), vectorized=True)
            x[k+1,:] = sol.y[:,-1]

        return x, u


if __name__ == "__main__":
    # Create FlexibleArm instance
    n_seg = 3
    fa = FlexibleArm(n_seg)

    # Sample a random configuration
    q = pin.randomConfiguration(fa.model)
    # fa.visualize(q)

    # Simulate
    q = np.zeros((fa.nq, 1))
    dq = np.zeros_like(q)
    x0 = np.vstack((q, dq))

    # controller = DummyController()
    controller = PDController(Kp=150, Kd=1, q_ref=np.array([np.pi/4]))

    ts = 0.01
    n_iter = 150

    sim = Simulator(fa, controller, 'rk45')
    x, u = sim.simulate(x0.flatten(), ts, 150)

    # Parse joint positions
    q = x[:,:fa.nq]

    # Animate simulated motion
    anim = Animator(fa, q).animate()