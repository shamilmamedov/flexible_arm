import casadi as cs
import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from numpy.random import multivariate_normal

from envs.flexible_arm_3dof import SymbolicFlexibleArm3DOF, get_rest_configuration
from integrator import RK4
from utils.utils import StateType

# Measurement noise covairance parameters
R_Q = [3e-6] * 3
R_DQ = [2e-3] * 3
R_PEE = [1e-4] * 3


@dataclass
class SimulatorOptions:
    """
    NOTE R is the covariance of the measurements error
    """

    rtol: float = 1e-8  # 1e-6, 1e-8
    atol: float = 1e-10  # 1e-8, 1e-10
    R: np.ndarray = np.diag([*R_Q, *R_DQ, *R_PEE])
    # R: np.ndarray = np.zeros((9, 9))
    contr_input_states: StateType = StateType.REAL
    dt: float = 0.01
    n_iter: float = 100


class Simulator:
    """Implements a simulator for FlexibleArm"""

    def __init__(
        self,
        robot,
        controller,
        integrator,
        estimator=None,
        opts: SimulatorOptions = SimulatorOptions(),
    ) -> None:
        if integrator in ["collocation", "cvodes"]:
            assert isinstance(robot, SymbolicFlexibleArm3DOF)

        self.robot = robot
        self.controller = controller
        self.integrator = integrator
        self.estimator = estimator
        self.opts = opts

        # Sanity checks
        if self.opts.contr_input_states is StateType.ESTIMATED:
            assert estimator is not None

        # Create an integrator for collocation method
        if self.integrator in ["collocation", "cvodes"]:
            dae = {"x": self.robot.x, "p": self.robot.u, "ode": self.robot.rhs}
            if self.integrator == "collocation":
                opts = {
                    "t0": 0,
                    "tf": self.opts.dt,
                    "number_of_finite_elements": 3,
                    "simplify": True,
                    "collocation_scheme": "radau",
                    "rootfinder": "fast_newton",
                    "expand": True,
                    "interpolation_order": 3,
                }
            else:
                opts = {
                    "t0": 0,
                    "tf": self.opts.dt,
                    "abstol": self.opts.atol,
                    "reltol": self.opts.rtol,
                    "nonlinear_solver_iteration": "newton",
                    "expand": True,
                    "linear_multistep_method": "bdf",
                }
            I = cs.integrator("I", self.integrator, dae, opts)
            x_next = I(x0=self.robot.x, p=self.robot.u)["xf"]
            self.F = cs.Function("F", [self.robot.x, self.robot.u], [x_next])
        self.x = None
        self.u = None
        self.y = None
        self.nx_est = None
        self.x_hat = None
        self.k = 0

    def reset(self, x0, n_iter: int = None) -> np.ndarray:
        self.k = 0
        if n_iter is not None:
            self.opts.n_iter = n_iter

        # Reset the estimator
        if self.estimator is not None:
            qa_0 = x0[self.robot.qa_idx]
            self.nx_est = self.estimator.model.nx
            self.estimator.reset(qa_0)

            # initialize datastructure
            self.x_hat = np.zeros((self.opts.n_iter + 1, self.nx_est))
        else:
            self.x_hat = None

        # Create arrays for states, controls and outputs
        self.x = np.zeros((self.opts.n_iter + 1, self.robot.nx))
        self.u = np.zeros((self.opts.n_iter, self.robot.nu))
        self.y = np.zeros((self.opts.n_iter + 1, self.robot.ny))

        # Initialize states and outputs
        self.x[0, :] = x0
        self.y[0, :] = self.robot.output(
            self.x[[0], :].T
        ).flatten() + multivariate_normal(np.zeros(self.robot.ny), self.opts.R)

        # Estimate the first state (Warmup)
        if self.estimator is not None:
            for _ in range(50):
                self.x_hat[0, :] = self.estimator.estimate(self.y[[0], :].T).flatten()

        # Return initial state for controller
        if self.opts.contr_input_states is StateType.REAL:
            qk = self.x[[self.k], : self.robot.nq].T
            dqk = self.x[[self.k], self.robot.nq :].T
        elif self.opts.contr_input_states is StateType.ESTIMATED:
            qk = self.x_hat[[self.k], : int(self.nx_est / 2)].T
            dqk = self.x_hat[[self.k], int(self.nx_est / 2) :].T
        else:
            raise NotImplementedError

        return np.vstack((qk, dqk))

    @staticmethod
    def ode_wrapper(t, x, robot, tau):
        """Wraps ode of the FlexibleArm to match scipy notation"""
        return robot.ode(x, tau)

    def integrator_step(self, x, u, dt) -> np.ndarray:
        """Implements one step of the simulation

        :parameter x: [nx x 1] (initial) state
        :parameter u: [nu x 1] control action
        :parameter dt: step size of the integrator
        """
        if self.integrator in ["RK45", "LSODA"]:
            sol = solve_ivp(
                self.ode_wrapper,
                [0, dt],
                x.flatten(),
                args=(self.robot, u),
                vectorized=True,
                method=self.integrator,
                rtol=self.opts.rtol,
                atol=self.opts.atol,
            )
            x_next = sol.y[:, -1]
        elif self.integrator == "RK4":
            x_next = RK4(x, u, self.robot.ode, dt, n=5).flatten()
        elif self.integrator in ["collocation", "cvodes"]:
            x_next = np.array(self.F(x, u)).flatten()
        else:
            raise ValueError
        return x_next

    def step(self, input_tau: np.ndarray):
        self.u[[self.k], :] = input_tau

        # Perform an integration step
        x_next = self.integrator_step(
            self.x[[self.k], :].T, self.u[[self.k], :].T, self.opts.dt
        )
        self.x[self.k + 1, :] = x_next

        # Compute output of the system
        self.y[self.k + 1, :] = self.robot.output(
            self.x[[self.k + 1], :].T
        ).flatten() + multivariate_normal(np.zeros(self.robot.ny), self.opts.R)

        # Estimate states if needed
        if self.estimator is not None:
            self.x_hat[self.k + 1, :] = self.estimator.estimate(
                self.y[[self.k + 1], :].T, self.u[self.k, :]
            ).flatten()

        # Increement the counter
        self.k += 1

        # Compute control action
        if self.opts.contr_input_states is StateType.REAL:
            qk = self.x[[self.k], : self.robot.nq].T
            dqk = self.x[[self.k], self.robot.nq :].T
        elif self.opts.contr_input_states is StateType.ESTIMATED:
            qk = self.x_hat[[self.k], : int(self.nx_est / 2)].T
            dqk = self.x_hat[[self.k], int(self.nx_est / 2) :].T
        else:
            raise NotImplementedError

        return np.vstack((qk, dqk))

    def simulate(self, x0, n_iter: int = None):
        state = self.reset(x0, n_iter)
        nq = int(state.shape[0] / 2)
        qk, dqk = state[0:nq, :], state[nq:, :]

        for k in range(self.opts.n_iter):
            tau = self.controller.compute_torques(
                qk, dqk, t=self.opts.dt * k, y=self.y[k, :].T
            )
            state = self.step(input_tau=tau)
            qk, dqk = state[0:nq, :], state[nq:, :]

        return self.x, self.u, self.y, self.x_hat


if __name__ == "__main__":
    opts = SimulatorOptions()
    print(opts.R)
