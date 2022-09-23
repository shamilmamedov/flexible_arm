import matplotlib
import casadi as cs
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from flexible_arm import FlexibleArm
from animation import Animator, Panda3dAnimator
from controller import DummyController, PDController


def RK4(x, u, ode, ts, n):
    """ Numerical RK4 integrator
    """
    h = ts/n
    x_next = x
    for _ in range(n):
        k1 = ode(x_next, u)
        k2 = ode(x_next + h*k1/2, u)
        k3 = ode(x_next + h*k2/2, u)
        k4 = ode(x_next + h*k3, u)
        x_next = x_next + h/6*(k1 + 2*k2 + 2*k3 + k4)
    return x_next


def symbolic_RK4(x, u, ode, n=4):
    """ Creates a symbolic RK4 integrator for
    a given dynamic system
    :parameter x: symbolic vector of states
    :parameter u: symbolic vector of inputs
    :parameter ode: ode of the system
    :parameter n: number of step for RK4 to take
    :return F_rk4: symbolic RK4 integrator
    """
    ts_sym = cs.MX.sym('ts')
    h = ts_sym/n
    x_next = x
    for _ in range(n):
        k1 = ode(x_next, u)
        k2 = ode(x_next + h*k1/2, u)
        k3 = ode(x_next + h*k2/2, u)
        k4 = ode(x_next + h*k3, u)
        x_next = x_next + h/6*(k1 + 2*k2 + 2*k3 + k4)

    F = cs.Function('F', [x, u, ts_sym], [x_next],
                    ['x', 'u', 'ts'], ['x_next'])

    A = cs.jacobian(x_next, x)
    dF_dx = cs.Function('dF_dx', [x, u, ts_sym], [A],
                        ['x', 'u', 'ts'], ['A'])
    return F, dF_dx


class Simulator:
    """ Implements a simulator for FlexibleArm
    """

    def __init__(self, robot, controller, integrator, estimator=None, rtol=1e-6, atol=1e-8) -> None:
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

    def simulate(self, x0, ts, n_iter):
        if self.estimator is not None:
            x_hat = np.zeros((n_iter+1, self.robot.nx))
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
                    x_hat[k, :] = self.estimator.estimate(y[[k], :].T).flatten()
                else:
                    x_hat[k, :] = self.estimator.estimate(
                                  y[[k], :].T, u[k-1, :]).flatten()

            tau = self.controller.compute_torques(qk, dqk)
            u[[k], :] = tau
            
            if self.integrator in ['RK45', 'LSODA']:
                sol = solve_ivp(self.ode_wrapper, [0, ts], x[k, :], args=(self.robot, tau),
                                vectorized=False, rtol=self.rtol, atol=self.atol, method=self.integrator)
                x_next = sol.y[:, -1]
            elif self.integrator == 'RK4':
                x_next = RK4(x[[k], :].T, tau.T, self.robot.ode, ts, n=5).flatten()
            else:
                raise ValueError
            x[k+1, :] = x_next
            y[k+1, :] = self.robot.output(x[[k+1], :].T).flatten()
        

        return x, u, y, x_hat


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
    q = x[::10, :fa.nq]

    _, ax = plt.subplots()
    ax.plot(t[::10], q[:, 0])
    # plt.show()

    # Animate simulated motion
    # anim = Animator(fa, q).play()

    urdf_path = 'models/one_dof/five_segments/flexible_arm_1dof_5s.urdf'
    animator = Panda3dAnimator(urdf_path, 0.01, q).play(3)
