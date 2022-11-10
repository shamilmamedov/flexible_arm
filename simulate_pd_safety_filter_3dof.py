from copy import deepcopy

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
from safty_filter_3dof import SafetyFilter3Dof, SafetyFilter3dofOptions, get_safe_controller_class
from utils import print_timings


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


def plot_real_states_vs_estimate(t, x, x_hat):
    _, ax = plt.subplots()
    ax.plot(t, x)
    ax.plot(t, x_hat, ls='--')
    plt.show()


if __name__ == "__main__":
    # Simulation parametes
    ts = 0.01
    n_iter = 500

    # Create FlexibleArm instance
    n_seg = 1
    n_seg_mpc = 1

    fa_ld = FlexibleArm3DOF(n_seg_mpc)
    fa = FlexibleArm3DOF(n_seg)
    fa_sym_ld = SymbolicFlexibleArm3DOF(n_seg_mpc)
    fa_sym = SymbolicFlexibleArm3DOF(n_seg)

    # Initial state
    q = np.zeros((fa.nq, 1))
    dq = np.zeros_like(q)
    x0 = np.vstack((q, dq))

    # Reference
    q_ref = np.zeros((fa.nq, 1))
    q_ref[0] += 2.
    q_ref[1] += 0.5
    q_ref[1 + n_seg + 1] += 1

    # MPC has other model states
    q0_mpc = np.zeros((fa_ld.nq, 1))
    dq0_mpc = np.zeros_like(q0_mpc)
    x0_mpc = np.vstack((q0_mpc, dq0_mpc))
    _, qee0 = fa_ld.fk_ee(q0_mpc)

    # saftey filter
    # Create mpc options and controller
    safety_options = SafetyFilter3dofOptions(n_seg=n_seg_mpc)
    safety_filter = SafetyFilter3Dof(model=fa_sym_ld, model_nonsymbolic=fa_ld, x0=x0_mpc, x0_ee=qee0,
                                     options=safety_options)
    u_ref = np.zeros((fa_sym_ld.nu, 1))  # u_ref could be changed to some known value along the trajectory

    # Controller
    # controller = DummyController()
    PDController3DofSafe = get_safe_controller_class(PDController3Dof, safety_filter=safety_filter)
    C = PDController3DofSafe(Kp=(0.2, 3, 3), Kd=(0.025, 0.025, 0.025), n_seg=n_seg, q_ref=q_ref)

    # Estimator
    # E = None
    est_model = SymbolicFlexibleArm3DOF(3, dt=ts)
    P0 = 0.01 * np.ones((est_model.nx, est_model.nx))
    q_q, q_dq = [1e-2] * est_model.nq, [1e-1] * est_model.nq
    Q = np.diag([*q_q, *q_dq])
    r_q, r_dq, r_pee = [3e-4] * 3, [6e-2] * 3, [1e-2] * 3
    R = np.diag([*r_q, *r_dq, *r_pee])
    E = ExtendedKalmanFilter(est_model, x0, P0, Q, R)

    # Simulate
    integrator = 'LSODA'
    sim = Simulator(fa, C, integrator, None)
    x, u, y, x_hat = sim.simulate(x0.flatten(), n_iter)
    t = np.arange(0, n_iter + 1) * ts

    # Print timing
    t_mean, t_std, t_min, t_max = safety_filter.get_timing_statistics()
    print_timings(t_mean, t_std, t_min, t_max)
    t_mean, t_std, t_min, t_max = safety_filter.get_timing_statistics(mode=1)
    print_timings(t_mean, t_std, t_min, t_max, name="Total")

    # Parse joint positions and plot active joints positions
    n_skip = 2
    q = x[::n_skip, :fa.nq]

    # plot_real_states_vs_estimate(t, x, x_hat)
    # plot_joint_positions(t[::10], q, n_seg, q_ref)

    # Animate simulated motion
    # anim = Animator(fa, q).play()

    #urdf_path = 'models/three_dof/five_segments/flexible_arm_3dof_5s.urdf'
    urdf_path = 'models/three_dof/one_segments/flexible_arm_3dof_1s.urdf'
    animator = Panda3dAnimator(urdf_path, 0.01, q).play(30)
