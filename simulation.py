#!/usr/bin/env python3

import matplotlib
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from flexible_arm import FlexibleArm
from animation import Animator, Panda3dAnimator
from controller import DummyController, PDController


class Simulator:
    """ Implements a simulator for FlexibleArm
    """
    def __init__(self, robot, controller, integrator, rtol=1e-6, atol=1e-8) -> None:
        self.robot = robot
        self.controller = controller
        self.integrator = integrator
        self.rtol = rtol
        self.atol = atol

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
            
            tau = self.controller.compute_torques(qk, dqk)
            u[[k],:] = tau

            sol = solve_ivp(self.ode_wrapper, [0, ts], x[k,:], args=(self.robot, tau),
                            vectorized=True, rtol=self.rtol, atol=self.atol, method=self.integrator)
            x[k+1,:] = sol.y[:,-1]

        return x, u


if __name__ == "__main__":
    # Create FlexibleArm instance
    n_seg = 5
    fa = FlexibleArm(n_seg)

    # Sample a random configuration
    q = pin.randomConfiguration(fa.model)
    # fa.visualize(q)

    # Simulate
    q = np.zeros((fa.nq, 1))
    dq = np.zeros_like(q)
    x0 = np.vstack((q, dq))

    # controller = DummyController()
    controller = PDController(Kp=10, Kd=0.25, q_ref=np.array([np.pi/8]))

    ts = 0.001
    n_iter = 1000

    sim = Simulator(fa, controller, 'LSODA')
    x, u = sim.simulate(x0.flatten(), ts, n_iter)
    t = np.arange(0, n_iter+1)*ts

    # Parse joint positions
    q = x[::10,:fa.nq]

    _, ax = plt.subplots()
    ax.plot(t[::10], q[:,0])
    # plt.show()
    

    # Animate simulated motion
    # anim = Animator(fa, q).play()

    urdf_path = 'models/one_dof/five_segments/flexible_arm_1dof_5s.urdf'
    animator = Panda3dAnimator(urdf_path, 0.01, q).play(3)