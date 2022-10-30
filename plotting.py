import numpy as np
import matplotlib.pyplot as plt


def plot_controls(t: np.ndarray, u: np.ndarray):
    u_lbls = [f'u_{k} [Nm]' for k in range(1,4)]
    fig, axs = plt.subplots(3,1, sharex=True, figsize=(6,8))

    for k, ax in enumerate(axs.reshape(-1)):
        ax.plot(t, u[:,k])
        ax.set_ylabel(u_lbls[k])
        ax.grid()
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


def plot_joint_velocities():
    pass