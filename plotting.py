import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def latexify(fig_width=None, fig_height=None):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.
    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    if fig_width is None:
        fig_width = 5  # width in inches

    if fig_height is None:
        golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches


    params = {#'backend': 'ps',
              'text.latex.preamble': r"\usepackage{gensymb} \usepackage{amsmath}",
              'axes.labelsize': 10, # fontsize for x and y labels (was 10)
              'axes.titlesize': 10,
              'legend.fontsize': 8,
              'xtick.labelsize': 10,
              'ytick.labelsize': 10,
              'text.usetex': True,
              'figure.figsize': [fig_width, fig_height],
              'font.family': 'serif'
    }

    matplotlib.rcParams.update(params)


def plot_controls(t: np.ndarray, u: np.ndarray, u_ref: np.ndarray = None):
    """
    :parameter t: time
    :parameter u: [n x 3] vector of control inputs
    :parameter u_ref: [3] or [3 x 1] vector of reference input
    """
    u_lbls = [fr'$\tau_{k}$ [Nm]' for k in range(1,4)]
    fig, axs = plt.subplots(3,1, sharex=True, figsize=(6,8))

    for k, ax in enumerate(axs.reshape(-1)):
        ax.plot(t, u[:,k])
        if u_ref is not None:
            ax.axhline(u_ref[k], ls='--', color='red')
        ax.set_ylabel(u_lbls[k])
        ax.grid(alpha=0.5)
    axs[2].set_xlabel('t [s]')
    plt.tight_layout()
    plt.show()


def plot_measurements(t: np.ndarray, y: np.ndarray, pee_ref: np.ndarray = None):
    # Parse measurements
    qa = y[:,:3]
    dqa = y[:,3:6]
    pee = y[:,6:9]

    qa_lbls = [fr'$qa_{k}$ [rad]' for k in range(1,4)]
    _, axs_q = plt.subplots(3,1, sharex=True, figsize=(6,8))
    for k, ax in enumerate(axs_q.reshape(-1)):
        ax.plot(t, qa[:,k])
        ax.set_ylabel(qa_lbls[k])
        ax.grid(alpha=0.5)
    axs_q[2].set_xlabel('t [s]')
    plt.tight_layout()

    dqa_lbls = [fr'$\dot qa_{k}$ [rad/s]' for k in range(1,4)]
    _, axs_dq = plt.subplots(3,1, sharex=True, figsize=(6,8))
    for k, ax in enumerate(axs_dq.reshape(-1)):
        ax.plot(t, dqa[:,k])
        ax.set_ylabel(dqa_lbls[k])
        ax.grid(alpha=0.5)
    plt.tight_layout()

    pee_lbls = [r'$p_{ee,x}$ [m]', r'$p_{ee,y}$ [m]', r'$p_{ee,z}$ [m]']
    _, axs_pee = plt.subplots(3,1, sharex=True, figsize=(6,8))
    for k, ax in enumerate(axs_pee.reshape(-1)):
        ax.plot(t, pee[:,k])
        if pee_ref is not None:
            if pee_ref.size == 3:
                ax.axhline(pee_ref[k], ls='--', color='red')
            else:
                ax.plot(t, pee_ref[:,k], ls='--', color='red')
        ax.set_ylabel(pee_lbls[k])
        ax.grid(alpha=0.5)
    plt.tight_layout()

    plt.show()


def plot_ee_positions(t: np.ndarray, pee: np.ndarray, pee_ref: np.ndarray = None):
    pee_lbls = [r'$p_{ee,x}$ [m]', r'$p_{ee,y}$ [m]', r'$p_{ee,z}$ [m]']
    _, axs_pee = plt.subplots(3,1, sharex=True, figsize=(6,8))
    for k, ax in enumerate(axs_pee.reshape(-1)):
        ax.plot(t, pee[:,k])
        if pee_ref is not None:
            ax.plot(t, pee_ref[:, k], ls='--', color='red')
        ax.set_ylabel(pee_lbls[k])
        ax.grid(alpha=0.5)
    plt.tight_layout()

    plt.show()


def plot_joint_positions(t: np.ndarray, q: np.ndarray, 
                         n_seg: int, q_ref: np.ndarray = None):
    """ Plots positions of the active joints
    """
    qa_idx = [0, 1, 1 + (n_seg + 1)]
    qa_lbls = [r"$q_{a 1}$", r"$q_{a 2}$", r"$q_{a 3}$"]

    _, axs = plt.subplots(3, 1, sharex=True)
    for k, (ax, qk) in enumerate(zip(axs.T, q[:, [qa_idx]].T)):
        ax.plot(t, qk.flatten())
        if q_ref is not None:
            ax.axhline(q_ref[qa_idx[k]], ls='--')
        ax.set_ylabel(qa_lbls[k])
        ax.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_real_states_vs_estimate(t: np.ndarray, x: np.ndarray, 
                                 x_hat: np.ndarray, n_seg_sim: int,
                                 n_seg_est: int):
    """
    Compares real and estimated states. 
    NOTE the comparison is valid if the discretization of the
    simulation model and of the estimator model match. If the
    dimensions of the estimation and simulation mdoels do 
    not match then only active joint positions and 
    velocities are compared and passive joint states are
    plotted alone
    """
    assert n_seg_sim >= n_seg_est

    # Parse states into active and passive
    nq_est = 3 + 2*n_seg_est
    nq_sim = 3 + 2*n_seg_sim
    idxs_est = set(np.arange(0, nq_est))
    idxs_sim = set(np.arange(0, nq_sim))
    qa_idxs_est = [0, 1, 2 + n_seg_est]
    qa_idxs_sim = [0, 1, 2 + n_seg_sim]
    qp_idxs_est = list(idxs_est.difference(set(qa_idxs_est)))
    qp_idxs_sim = list(idxs_sim.difference(set(qa_idxs_sim)))

    dqa_idxs_est = [nq_est + k for k in qa_idxs_est]
    dqa_idxs_sim = [nq_sim + k for k in qa_idxs_sim]
    dqp_idxs_est = [nq_est + k for k in qp_idxs_est]
    dqp_idxs_sim = [nq_sim + k for k in qp_idxs_sim]

    # Plot active joint positions
    qa_lbls = [r"$q_{a 1}$", r"$q_{a 2}$", r"$q_{a 3}$"]
    fig_qa, axs_qa = plt.subplots(3, 1, sharex=True, figsize=(6,8))
    for k, (ax, qk, qk_hat) in enumerate(zip(axs_qa.T, 
                        x[:, [qa_idxs_sim]].T, x_hat[:, [qa_idxs_est]].T)):
        ax.plot(t, qk.flatten())
        ax.plot(t, qk_hat.flatten(), ls='--')
        ax.set_ylabel(qa_lbls[k])
        ax.grid(alpha=0.5)
    fig_qa.suptitle('active joint positions')
    fig_qa.tight_layout()

    # # Plot passive joint positions
    qp1_idxs_est = qp_idxs_est[:len(qp_idxs_est)//2]
    qp2_idxs_est = qp_idxs_est[len(qp_idxs_est)//2:]
    fig_qp, axs_qp = plt.subplots(2,1, sharex=True, figsize=(6,8))
    for k, (ax, idxs) in enumerate(zip(axs_qp.T, [qp1_idxs_est, qp2_idxs_est])):
        for (qk, qk_hat) in zip(x[:, [idxs]].T, x_hat[:, [idxs]].T):
            ax.plot(t, qk_hat.flatten(), ls='--')
            if n_seg_est == n_seg_sim:
                ax.plot(t, qk.flatten())    
            ax.grid(alpha=0.5)
    fig_qp.suptitle('passive joint positions')
    fig_qp.tight_layout()

    # Plot active joint velocities
    dqa_lbls = [r"$\dot q_{a 1}$", r"$\dot q_{a 2}$", r"$\dot q_{a 3}$"]
    fig_dqa, axs_dqa = plt.subplots(3, 1,  sharex=True, figsize=(6,8))
    for k, (ax, dqk, dqk_hat) in enumerate(zip(axs_dqa.T, 
                        x[:, [dqa_idxs_sim]].T, x_hat[:, [dqa_idxs_est]].T)):
        ax.plot(t, dqk.flatten())
        ax.plot(t, dqk_hat.flatten(), ls='--')
        ax.set_ylabel(dqa_lbls[k])
        ax.grid(alpha=0.5)
    fig_dqa.suptitle('active joint velocities')
    fig_dqa.tight_layout()

    # Passive joint velocities
    dqp1_idxs_est = dqp_idxs_est[:len(dqp_idxs_est)//2]
    dqp2_idxs_est = dqp_idxs_est[len(dqp_idxs_est)//2:]
    fig_dqp, axs_dqp = plt.subplots(2,1, sharex=True, figsize=(6,8))
    for k, (ax, idxs) in enumerate(zip(axs_dqp.T, 
                            [dqp1_idxs_est, dqp2_idxs_est])):
        for (dqk, dqk_hat) in zip(x[:, [idxs]].T, x_hat[:, [idxs]].T):
            if n_seg_est == n_seg_sim:
                ax.plot(t, dqk.flatten())
            ax.plot(t, dqk_hat.flatten(), ls='--')
            ax.grid(alpha=0.5)
    fig_dqp.suptitle('passive joint velocities')
    fig_dqp.tight_layout()

    plt.show()


def plot_joint_velocities(t: np.ndarray, dq: np.ndarray, 
                         n_seg: int, dq_ref: np.ndarray = None):
    """ Plots positions of the active joints
    """
    dqa_idx = [0, 1, 1 + (n_seg + 1)]
    dqa_lbls = [r"$\dot q_{a 1}$", r"$\dot q_{a 2}$", r"$\dot q_{a 3}$"]

    _, axs = plt.subplots(3, 1, sharex=True)
    for k, (ax, dqk) in enumerate(zip(axs.T, dq[:, [dqa_idx]].T)):
        ax.plot(t, dqk.flatten())
        if dq_ref is not None:
            ax.axhline(dq_ref[dqa_idx[k]], ls='--')
        ax.set_ylabel(dqa_lbls[k])
        ax.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()