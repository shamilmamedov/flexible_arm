import matplotlib
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from flexible_arm_3dof import FlexibleArm3DOF, SymbolicFlexibleArm3DOF
from animation import Animator, Panda3dAnimator
from controller import DummyController, PDController3Dof
from simulation import Simulator
from estimator import ExtendedKalmanFilter


def plot_joint_positions(t, q, n_seg: int, q_ref: np.ndarray = None):
    """ Plots positions of the active joints
    """
    qa_idx = [0, 1, 1 + (n_seg + 1)]
    qa_lbls = [r"$q_{a 1}$", r"$q_{a 2}$", r"$q_{a 3}$"]

    _, axs = plt.subplots(3, 1)
    for k, (ax, qk) in enumerate(zip(axs.T, q[:,[qa_idx]].T)):
        ax.plot(t, qk.flatten())
        if q_ref is not None:
            ax.axhline(q_ref[qa_idx[k]], ls='--')
        ax.set_ylabel(qa_lbls[k])
        ax.grid(alpha=0.5)
        
    plt.show()

def plot_real_states_vs_estimate(t, x, x_hat):
    _, ax = plt.subplots()
    ax.plot(t, x)
    ax.plot(t, x_hat, ls='--')
    plt.show()


if __name__ == "__main__":
    # Simulation parametes
    ts = 0.001
    n_iter = 500

    # Create FlexibleArm instance
    n_seg = 3
    fa = FlexibleArm3DOF(n_seg)

    # Initial state
    q = np.zeros((fa.nq, 1))
    dq = np.zeros_like(q)
    x0 = np.vstack((q, dq))

    # Reference
    q_ref = np.zeros((fa.nq, 1))
    q_ref[0] += 1.
    q_ref[1] += 0.5
    q_ref[1 + n_seg + 1] += 1

    # Controller
    # controller = DummyController()
    C = PDController3Dof(Kp=(40, 40, 40), Kd=(0.25, 0.25, 0.25),
                         n_seg=n_seg, q_ref=q_ref)

    # Estimator
    # E = None
    est_model = SymbolicFlexibleArm3DOF(3, ts=ts)
    P0 = 0.01*np.ones((est_model.nx, est_model.nx))
    q_q, q_dq = [1e-2]*est_model.nq, [1e-1]*est_model.nq
    Q = np.diag([*q_q, *q_dq])
    r_q, r_dq, r_pee = [3e-4]*3, [6e-2]*3, [1e-2]*3
    R = np.diag([*r_q, *r_dq, *r_pee])
    E = ExtendedKalmanFilter(est_model, x0, P0, Q, R)

    # Simulate
    integrator = 'LSODA'
    sim = Simulator(fa, C, integrator, E)
    x, u, x_hat = sim.simulate(x0.flatten(), ts, n_iter)
    t = np.arange(0, n_iter + 1) * ts

    # Parse joint positions and plot active joints positions
    q = x[::10, :fa.nq]

    plot_real_states_vs_estimate(t, x, x_hat)
    plot_joint_positions(t[::10], q, n_seg, q_ref)


    # Animate simulated motion
    # anim = Animator(fa, q).play()

    urdf_path = 'models/three_dof/three_segments/flexible_arm_3dof_3s.urdf'
    animator = Panda3dAnimator(urdf_path, 0.01, q).play(3)
