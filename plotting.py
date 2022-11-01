import numpy as np
import matplotlib.pyplot as plt


def plot_controls(t: np.ndarray, u: np.ndarray):
    u_lbls = [f'u_{k} [Nm]' for k in range(1,4)]
    fig, axs = plt.subplots(3,1, sharex=True, figsize=(6,8))

    for k, ax in enumerate(axs.reshape(-1)):
        ax.plot(t, u[:,k])
        ax.set_ylabel(u_lbls[k])
        ax.grid(alpha=0.5)
    axs[2].set_xlabel('t [s]')
    plt.tight_layout()
    plt.show()


def plot_measurements(t: np.ndarray, y: np.ndarray):
    # Parse measurements
    qa = y[:,:3]
    dqa = y[:,3:6]
    pee = y[:,6:9]

    qa_lbls = [f'qa_{k} [rad]' for k in range(1,4)]
    _, axs_q = plt.subplots(3,1, sharex=True, figsize=(6,8))
    for k, ax in enumerate(axs_q.reshape(-1)):
        ax.plot(t, qa[:,k])
        ax.set_ylabel(qa_lbls[k])
        ax.grid(alpha=0.5)
    axs_q[2].set_xlabel('t [s]')
    plt.tight_layout()

    dqa_lbls = [f'dqa_{k} [rad/s]' for k in range(1,4)]
    _, axs_dq = plt.subplots(3,1, sharex=True, figsize=(6,8))
    for k, ax in enumerate(axs_dq.reshape(-1)):
        ax.plot(t, dqa[:,k])
        ax.set_ylabel(dqa_lbls[k])
        ax.grid(alpha=0.5)
    plt.tight_layout()

    pee_lbls = ['pee_x [m]', 'pee_y [m]', 'pee_z [m]']
    _, axs_pee = plt.subplots(3,1, sharex=True, figsize=(6,8))
    for k, ax in enumerate(axs_pee.reshape(-1)):
        ax.plot(t, pee[:,k])
        ax.set_ylabel(pee_lbls[k])
        ax.grid(alpha=0.5)
    plt.tight_layout()

    plt.show()


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


def plot_joint_velocities():
    pass