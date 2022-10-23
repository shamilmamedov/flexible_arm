import casadi as cs
import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from numpy.random import multivariate_normal

from flexible_arm_3dof import SymbolicFlexibleArm3DOF
from integrator import RK4

R_Q = [3e-6]*3
R_DQ = [5e-3]*3
R_PEE = [5e-4]*3

@dataclass
class SimulatorOptions:
    """
    NOTE R is the covariance of the measurements error
    """
    rtol: float = 1e-3
    atol: float = 1e-6
    R: np.ndarray = np.diag([*R_Q, *R_DQ, *R_PEE])
    # R: np.ndarray = np.zeros((9,9))
    contr_input_states: str = 'real'


class Simulator:
    """ Implements a simulator for FlexibleArm
    """

    def __init__(self, robot, controller, integrator, estimator=None, 
                 opts: SimulatorOptions = SimulatorOptions()) -> None:
        if integrator in ['collocation', 'cvodes']:
            assert isinstance(robot, SymbolicFlexibleArm3DOF)

        self.robot = robot
        self.controller = controller
        self.integrator = integrator
        self.estimator = estimator
        self.opts = opts

        # Sanity checks
        if self.opts.contr_input_states == 'estimated':
            assert(estimator is not None)

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
                            vectorized=True, method=self.integrator, 
                            rtol=self.opts.rtol, atol=self.opts.atol)
            x_next = sol.y[:, -1]
        elif self.integrator == 'RK4':
            x_next = RK4(x, u, self.robot.ode, dt, n=5).flatten()
        elif self.integrator in ['collocation', 'cvodes']:
            x_next = np.array(self.F(x, u)).flatten()
        else:
            raise ValueError
        return x_next

    def simulate(self, x0, dt, n_iter):
        # Create an integrator for collocation method
        if self.integrator in ['collocation', 'cvodes']:
            dae = {'x': self.robot.x, 'p': self.robot.u, 'ode': self.robot.rhs}
            if self.integrator == 'collocation':
                opts = {'t0': 0, 'tf': dt, 'number_of_finite_elements': 3, 
                        'simplify': True, 'collocation_scheme': 'radau',
                        'rootfinder':'fast_newton','expand': True, 
                        'interpolation_order': 3}
            else:
                opts = {'t0': 0, 'tf': dt, 'abstol':self.opts.atol, 'reltol':self.opts.rtol,
                        'nonlinear_solver_iteration': 'functional', 'expand': True,
                        'fsens_err_con': False, 'quad_err_con': False,
                        'linear_multistep_method': 'bdf'}
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
        y[0, :] = (self.robot.output(x[[0], :].T).flatten() + 
                   multivariate_normal(np.zeros(self.robot.ny), self.opts.R))
        for k in range(n_iter):
            if self.estimator is not None:
                if k == 0:
                    x_hat[k, :] = self.estimator.estimate(
                                    y[[k], :].T).flatten()
                else:
                    x_hat[k, :] = self.estimator.estimate(
                                    y[[k], :].T, u[k-1, :]).flatten()

            # Compute control action
            if self.opts.contr_input_states == 'real':
                qk = x[[k], :self.robot.nq].T
                dqk = x[[k], self.robot.nq:].T
            elif self.opts.contr_input_states == 'estimated':
                qk = x_hat[[k], :self.robot.nq].T
                dqk = x_hat[[k], self.robot.nq:].T
            tau = self.controller.compute_torques(qk, dqk, t=dt*k)
            u[[k], :] = tau

            # Perform an integration step
            x_next = self.step(x[[k], :].T,  u[[k], :].T, dt)
            x[k+1, :] = x_next

            # Compute output of the system
            y[k+1, :] = (self.robot.output(x[[k+1], :].T).flatten() + 
                         multivariate_normal(np.zeros(self.robot.ny), self.opts.R))

        return x, u, y, x_hat


if __name__ == "__main__":
    opts = SimulatorOptions()
    print(opts.R)
