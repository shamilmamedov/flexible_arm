import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import unit_impulse

from simulation import Simulator, SimulatorOptions
from controller import (PDController3Dof, FeedforwardController,
                        DummyController)
from envs.flexible_arm_3dof import (SymbolicFlexibleArm3DOF,
                               get_rest_configuration)
from animation import Panda3dAnimator
import plotting



# Simulation parameters
RTOL = 1e-10
ATOL = 1e-12
DT = 0.001
N_ITER = 3000

def smooth_impulse_signal(N: int = 100):
    t = np.linspace(0, 1, 101)

    _, ax = plt.subplots()
    ax.plot(t, np.sin(np.pi*t))
    plt.show()


def unit_step_antistep(shape, idx_range):
    out = np.zeros(shape)
    out[idx_range,:] = 1
    return out


def compare_different_discretizations(save_figure: bool = False, latexify: bool = False):
    """ Compares models with different discretization -- starting 
    from a rigid body model and until a fine grid of 10 discretization
    points -- in simulatoin. For every simulation the same integrator, 
    controller and initial state are chosen.   
    """

    # Create FlexibleArm instance
    n_segs = [0, 1, 2, 3, 5, 10]
    fas = [SymbolicFlexibleArm3DOF(n_seg) for n_seg in n_segs]

    # Initial state; the same for all models
    # active joint configurations
    # qa = np.array([np.pi / 2, np.pi / 10, -np.pi / 8])
    # qa = np.array([0., 2 * np.pi / 5, -np.pi / 3])
    qa = np.array([0., -np.pi/2, 0.])
    # Q = [get_rest_configuration(qa, n_seg) for n_seg in n_segs]
    Q = [np.zeros((m.nq, 1)) for m in fas]
    for q, m in zip(Q, fas):
        q[m.qa_idx,0] = qa
    dQ = [np.zeros_like(q) for q in Q]
    X0 = [np.vstack((q, dq)) for q, dq in zip(Q, dQ)]

    
    # Estimator
    E = None

    # Controller
    u = np.hstack((np.zeros((N_ITER,1)), 
                   10*unit_step_antistep((N_ITER,1), range(500,542)),
                   5*unit_step_antistep((N_ITER,1), range(50,80))))
    C = [FeedforwardController(u) for _ in n_segs]
    # C = DummyController(n_joints=3)
    # C = FeedforwardController(u)

    # Simulators
    sim_opts = SimulatorOptions(rtol=RTOL, atol=ATOL, R=np.zeros((9,9)), 
                                dt=DT, n_iter=N_ITER)
    integrator = 'cvodes'

    # Create simulators for RFEM models
    S = [Simulator(fa, c, integrator, E, sim_opts) for fa, c in zip(fas, C)]

    # Simulate REFEM models
    x_0s, u_0s, y_0s, _ = S[0].simulate(X0[0].flatten())
    x_1s, u_1s, y_1s, _ = S[1].simulate(X0[1].flatten())
    x_2s, u_2s, y_2s, _ = S[2].simulate(X0[2].flatten())
    x_3s, u_3s, y_3s, _ = S[3].simulate(X0[3].flatten())
    x_5s, u_5s, y_5s, _ = S[4].simulate(X0[4].flatten())
    x_10s, u_10s, y_10s, _ = S[5].simulate(X0[5].flatten())

    t = np.arange(0, N_ITER + 1) * DT

    # q_10s = x_10s[::10, :fas[-1].nq]
    # Panda3dAnimator(fas[-1].urdf_path, DT * 10, q_10s).play(3)

    # Find the norm of the difference betweem 10s and other models
    delta_0s = y_10s - y_0s
    delta_1s = y_10s - y_1s
    delta_2s = y_10s - y_2s
    delta_3s = y_10s - y_3s
    delta_5s = y_10s - y_5s

    norm_delta_0s = np.linalg.norm(delta_0s, axis=0).reshape(1,-1)
    norm_delta_1s = np.linalg.norm(delta_1s, axis=0).reshape(1,-1)
    norm_delta_2s = np.linalg.norm(delta_2s, axis=0).reshape(1,-1)
    norm_delta_3s = np.linalg.norm(delta_3s, axis=0).reshape(1,-1)
    norm_delta_5s = np.linalg.norm(delta_5s, axis=0).reshape(1,-1)

    norm_delta = np.vstack((norm_delta_0s, norm_delta_1s, norm_delta_2s,
                            norm_delta_3s, norm_delta_5s))

    cols = ['|delta qa_1|', '|delta qa_2|', '|delta qa_3|', 
            '|delta dqa_1|', '|delta dqa_2|', '|delta dqa_3|', 
            '|delta ee_x|', '|delta ee_y|', '|delta ee_z|']

    df = pd.DataFrame(norm_delta, columns=cols, index=[0, 1, 2, 3, 5])
    print(df.iloc[:,6:])


    # Plot outputs
    fig_width, fig_height = 6, 2
    if latexify:
        plotting.latexify(fig_width, fig_height)
    
    y_lbls = [r'$p_{\mathrm{ee},x}$ [m]', r'$p_{\mathrm{ee},y}$ [m]',
              r'$p_{\mathrm{ee},z}$ [m]']
    legends = [f'${n_seg}$ ' + '$\mathrm{seg}$' for n_seg in n_segs]
    fig, axs = plt.subplots(1, 2, sharex=True, figsize=(fig_width, fig_height))
    # y_lims = [[-0.05, 0.85], [0.23, 0.75]]
    t_ = [0, 2]
    lw = 0.75

    zoom_axis = 1
    xlim_zoom = [1.6, 1.83]
    ylim_zoom = [-0.08, -0.02]
    for k, ax in enumerate(axs.reshape(-1)):
        ax.plot(t, y_0s[:, 6+t_[k]], label=legends[0], lw=lw)
        ax.plot(t, y_1s[:, 6+t_[k]], label=legends[1], lw=lw)
        ax.plot(t, y_2s[:, 6+t_[k]], label=legends[2], lw=lw)
        ax.plot(t, y_3s[:, 6+t_[k]], label=legends[3], lw=lw)
        ax.plot(t, y_5s[:, 6+t_[k]], ls='--', label=legends[4], lw=lw)
        ax.plot(t, y_10s[:, 6+t_[k]], ls='-.', label=legends[5], lw=lw)
        if k == zoom_axis:
            ax.vlines(xlim_zoom[0], ylim_zoom[0], ylim_zoom[1], colors='k', lw=0.5)
            ax.vlines(xlim_zoom[1], ylim_zoom[0], ylim_zoom[1], colors='k', lw=0.5)
            ax.hlines(ylim_zoom[0], xlim_zoom[0], xlim_zoom[1], colors='k', lw=0.5)
            ax.hlines(ylim_zoom[1], xlim_zoom[0], xlim_zoom[1], colors='k', lw=0.5)
        ax.set_ylabel(y_lbls[t_[k]])
        ax.grid(alpha=0.5)
        ax.set_xlim([0, 3])
        # ax.set_ylim(y_lims[k])
        # ax.legend(ncol=2)
        ax.set_xlabel(r'$t$ [s]')
    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=6)
    plt.tight_layout()

    lw_zoom = 1.2
    fig_zoom, ax_zoom = plt.subplots()
    ax_zoom.plot(t, y_0s[:, 6+t_[zoom_axis]], lw=lw_zoom)
    ax_zoom.plot(t, y_1s[:, 6+t_[zoom_axis]], lw=lw_zoom)
    ax_zoom.plot(t, y_2s[:, 6+t_[zoom_axis]], lw=lw_zoom)
    ax_zoom.plot(t, y_3s[:, 6+t_[zoom_axis]], lw=lw_zoom)
    ax_zoom.plot(t, y_5s[:, 6+t_[zoom_axis]], ls='--', lw=lw_zoom)
    ax_zoom.plot(t, y_10s[:, 6+t_[zoom_axis]], ls='-.', lw=lw_zoom)
    ax_zoom.set_xlim(xlim_zoom)
    ax_zoom.set_ylim(ylim_zoom)
    ax_zoom.set_xticks([])
    ax_zoom.set_yticks([])
    plt.tight_layout()

    if save_figure:
        fig.savefig('figures_L4DC/discr.pdf', format='pdf', dpi=600, bbox_inches='tight')
        fig_zoom.savefig('figures_L4DC/discr_zoom.pdf', format='pdf', dpi=600, 
                         bbox_inches='tight', pad_inches=0, transparent=True)

    # # Plot controls
    # y_lbls = [r'$\tau$ [m]', r'$\tau$ [m]', r'$\tau$ [m]']
    # _, axs = plt.subplots(3, 1, sharex=True, figsize=(6, 7))
    # for k, ax in enumerate(axs.reshape(-1)):
    #     ax.plot(t[:-1], u_0s[:, k], label=legends[0])
    #     ax.plot(t[:-1], u_1s[:, k], label=legends[1])
    #     ax.plot(t[:-1], u_2s[:, k], label=legends[2])
    #     ax.plot(t[:-1], u_3s[:, k], label=legends[3])
    #     ax.plot(t[:-1], u_5s[:, k], ls='--', label=legends[4])
    #     ax.plot(t[:-1], u_10s[:, k], ls='-.', label=legends[5])
    #     ax.set_ylabel(y_lbls[k])
    #     # ax.set_xlim([0, 0.025])
    #     ax.grid(alpha=0.5)
    #     ax.legend()
    # ax.set_xlabel('t [s]')
    # plt.tight_layout()
    plt.show()
    
 
 
if __name__ == "__main__":
    compare_different_discretizations(save_figure=True, latexify=True)
    # smooth_impulse_signal()
    # sig = unit_step_antistep((N_ITER,1), range(50,100))

    # _, ax = plt.subplots()
    # ax.plot(sig)
    # plt.show()