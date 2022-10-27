import matplotlib
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from flexible_arm_3dof import (FlexibleArm3DOF, SymbolicFlexibleArm3DOF, 
                                get_rest_configuration)
from animation import Animator, Panda3dAnimator
from controller import DummyController, PDController3Dof
from simulation import Simulator, SimulatorOptions
from estimator import ExtendedKalmanFilter


def plot_joint_positions(t, q, n_seg: int, q_ref: np.ndarray = None):
    """ Plots positions of the active joints
    """
    qa_idx = [0, 1, 1 + (n_seg + 1)]
    qa_lbls = [r"$q_{a 1}$", r"$q_{a 2}$", r"$q_{a 3}$"]

    _, axs = plt.subplots(3, 1)
    for k, (ax, qk) in enumerate(zip(axs.T, q[:, [qa_idx]].T)):
        ax.plot(t, qk.flatten())
        if q_ref is not None:
            ax.axhline(q_ref[qa_idx[k]], ls='--')
        ax.set_ylabel(qa_lbls[k])
        ax.grid(alpha=0.5)
    plt.show()


def plot_output_measurements(t, y):
    # Parse measurements
    qa = y[:,:3]
    dqa = y[:,3:6]
    pee = y[:,6:9]

    qa_lbls = [f'qa_{k} [rad]' for k in range(1,4)]
    _, axs_q = plt.subplots(3,1, sharex=True, figsize=(6,8))
    for k, ax in enumerate(axs_q.reshape(-1)):
        ax.plot(t, qa[:,k])
        ax.set_ylabel(qa_lbls[k])
        ax.grid()
    axs_q[2].set_xlabel('t [s]')
    plt.tight_layout()

    dqa_lbls = [f'dqa_{k} [rad/s]' for k in range(1,4)]
    _, axs_dq = plt.subplots(3,1, sharex=True, figsize=(6,8))
    for k, ax in enumerate(axs_dq.reshape(-1)):
        ax.plot(t, dqa[:,k])
        ax.set_ylabel(dqa_lbls[k])
        ax.grid()
    plt.tight_layout()

    pee_lbls = ['pee_x [m]', 'pee_y [m]', 'pee_z [m]']
    _, axs_pee = plt.subplots(3,1, sharex=True, figsize=(6,8))
    for k, ax in enumerate(axs_pee.reshape(-1)):
        ax.plot(t, pee[:,k])
        ax.set_ylabel(pee_lbls[k])
        ax.grid()
    plt.tight_layout()

    plt.show()


def plot_controls(t, u):
    u_lbls = [f'tau_{k} [Nm]' for k in range(1,4)]
    _, axs_u = plt.subplots(3,1, sharex=True, figsize=(6,8))
    for k, ax in enumerate(axs_u.reshape(-1)):
        ax.plot(t, u[:,k])
        ax.set_ylabel(u_lbls[k])
        ax.grid()
    axs_u[2].set_xlabel('t [s]')
    plt.tight_layout()
    plt.show()


def plot_real_states_vs_estimate(t, x, x_hat, n_seg: int):
    nq = 3 + 2*n_seg
    idxs = set(np.arange(0, nq))
    qa_idxs = [0, 1, 2 + n_seg]
    qp_idxs = list(idxs.difference(set(qa_idxs)))

    dqa_idxs = [nq + k for k in qa_idxs]
    dqp_idxs = [nq + k for k in qp_idxs]

    _, ax_qa = plt.subplots()
    ax_qa.plot(t, x[:,qa_idxs])
    ax_qa.plot(t, x_hat[:,qa_idxs], ls='--')
    ax_qa.set_title('Active joint positions')
    plt.tight_layout()

    _, ax_qp = plt.subplots()
    ax_qp.plot(t, x[:, qp_idxs])
    ax_qp.plot(t, x_hat[:, qp_idxs], ls='--')
    ax_qp.set_title('Passive joint positions')
    plt.tight_layout()

    _, ax_dqa = plt.subplots()
    ax_dqa.plot(t, x[:,dqa_idxs])
    ax_dqa.plot(t, x_hat[:,dqa_idxs], ls='--')
    ax_dqa.set_title('Active joint velocities')
    plt.tight_layout()

    _, ax_dqp = plt.subplots()
    ax_dqp.plot(t, x[:,dqp_idxs])
    ax_dqp.plot(t, x_hat[:,dqp_idxs], ls='--')
    ax_dqp.set_title('Passive joint velocities')
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    # Simulation parametes
    dt = 0.001
    n_iter = 600

    # Create FlexibleArm instance
    n_seg = 3
    fa = SymbolicFlexibleArm3DOF(n_seg)

    # Initial state
    qa = np.zeros(3)
    q = get_rest_configuration(qa, n_seg)
    dq = np.zeros_like(q)
    x0 = np.concatenate((q,dq)).reshape(-1,1)

    # Reference
    qa_ref = np.array([2, 0.5, 1])
    q_ref = get_rest_configuration(qa_ref, n_seg)

    # Estimator
    # E = None
    n_seg_est = 3
    est_model = SymbolicFlexibleArm3DOF(n_seg_est, dt=dt)
    P0 = 0.01*np.ones((est_model.nx, est_model.nx))
    q_q, q_dq = [1e-2]*est_model.nq, [1e-1]*est_model.nq
    Q = np.diag([*q_q, *q_dq])
    r_q, r_dq, r_pee = [3e-4]*3, [6e-2]*3, [1e-2]*3
    R = np.diag([*r_q, *r_dq, *r_pee])

    q_est = get_rest_configuration(qa, n_seg_est)
    dq_est = np.zeros_like(q_est)
    x0_est = np.concatenate((q_est, dq_est)).reshape(-1,1)
    E = ExtendedKalmanFilter(est_model, x0_est, P0, Q, R)

    # Controller
    # controller = DummyController()
    C = PDController3Dof(Kp=(40, 8, 2), Kd=(1.5, 0.2, 0.1),
                         n_seg=n_seg_est, q_ref=q_ref)

    # Simulate
    opts = SimulatorOptions(contr_input_states='estimated')
    integrator = 'cvodes'
    sim = Simulator(fa, C, integrator, E, opts)
    x, u, y, x_hat = sim.simulate(x0.flatten(), dt, n_iter)
    t = np.arange(0, n_iter + 1) * dt

    # Parse joint positions and plot active joints positions
    n_skip = 10
    q = x[::n_skip, :fa.nq]

    # plot_real_states_vs_estimate(t, x, x_hat)
    # plot_joint_positions(t[::10], q, n_seg, q_ref)
    # plot_controls(t[:-1], u)
    plot_real_states_vs_estimate(t, x, x_hat, n_seg)
    # plot_joint_positions(t[::n_skip], q, n_seg, q_ref)
    # plot_output_measurements(t, y)

    # Animate simulated motion
    animator = Panda3dAnimator(fa.urdf_path, 0.01, q).play(3)

