import matplotlib
import casadi as cs
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from flexible_arm import FlexibleArm
from animation import Animator, Panda3dAnimator
from controller import DummyController, PDController
from flexible_arm_3dof import SymbolicFlexibleArm3DOF
from integrator import RK4

class Simulator:
    """ Implements a simulator for FlexibleArm
    """

    def __init__(self, robot, controller, integrator, estimator=None, rtol=1e-6, atol=1e-8) -> None:
        if integrator == 'collocation':
            assert isinstance(robot, SymbolicFlexibleArm3DOF)

        self.robot = robot
        self.controller = controller
        self.integrator = integrator
        self.estimator = estimator
        self.rtol = rtol
        self.atol = atol

    @staticmethod
    def ode_wrapper(t, x, robot, tau):
        """ Wraps ode of the FlexibleArm to match scipy notation
        """
        return robot.ode(x, tau)

    def step(self, x, u, dt) -> np.ndarray:
        """ Implements one step of the simulation

        :parameter x: [nx x 1] (initial) state 
        :parameter u: [nu x 1] control action
        :parameter dt: step size of the integrator
        """
        if self.integrator in ['RK45', 'LSODA']:
            sol = solve_ivp(self.ode_wrapper, [0, dt], x.flatten(), args=(self.robot, u),
                            vectorized=True, rtol=self.rtol, atol=self.atol, method=self.integrator)
            x_next = sol.y[:, -1]
        elif self.integrator == 'RK4':
            x_next = RK4(x, u, self.robot.ode, dt, n=5).flatten()
        elif self.integrator == 'collocation':
            x_next = np.array(self.F(x, u)).flatten()
        else:
            raise ValueError
        return x_next

    def simulate(self, x0, dt, n_iter):
        # Create an integrator for collocation method
        if self.integrator == 'collocation':
            dae = {'x': self.robot.x, 'p': self.robot.u, 'ode': self.robot.rhs}
            opts = {'t0': 0, 'tf': dt, 'number_of_finite_elements': 5, 'simplify': True}
            I = cs.integrator('I', self.integrator, dae, opts)
            x_next = I(x0=self.robot.x, p=self.robot.u)["xf"]
            self.F = cs.Function('F', [self.robot.x, self.robot.u], [x_next])

        if self.estimator is not None:
            nx_est = np.shape(self.estimator.x_hat)[0]
            x_hat = np.zeros((n_iter+1, nx_est))
        else:
            x_hat = None

        x = np.zeros((n_iter+1, self.robot.nx))
        u = np.zeros((n_iter, self.robot.nu))
        y = np.zeros((n_iter+1, self.robot.ny))
        x[0, :] = x0
        y[0, :] = self.robot.output(x[[0], :].T).flatten()
        for k in range(n_iter):
            qk = x[[k], :self.robot.nq].T
            dqk = x[[k], self.robot.nq:].T

            if self.estimator is not None:
                if k == 0:
                    x_hat[k, :] = self.estimator.estimate(
                        y[[k], :].T).flatten()
                else:
                    x_hat[k, :] = self.estimator.estimate(
                        y[[k], :].T, u[k-1, :]).flatten()

            # Compute control action
            tau = self.controller.compute_torques(qk, dqk)
            u[[k], :] = tau

            # Perform an integration step
            x_next = self.step(x[[k], :].T,  u[[k], :].T, dt)
            x[k+1, :] = x_next

            # Compute output of the system
            y[k+1, :] = self.robot.output(x[[k+1], :].T).flatten()

        return x, u, y, x_hat


if __name__ == "__main__":
    pass
