from copy import deepcopy
import numpy as np
import pinocchio as pin

from controller import NNController
from estimator import ExtendedKalmanFilter
from utils import plot_result, print_timings, ControlMode
from flexible_arm_3dof import FlexibleArm3DOF, SymbolicFlexibleArm3DOF
from animation import Panda3dAnimator
from mpc_3dof import Mpc3dofOptions, Mpc3Dof
from simulation import Simulator

if __name__ == "__main__":
    # options
    control_mode = ControlMode.SET_POINT
    use_NN = True

    # Create FlexibleArm instance
    n_seg = 3
    n_seg_mpc = 3

    fa_ld = FlexibleArm3DOF(n_seg_mpc)
    fa_hd = FlexibleArm3DOF(n_seg)
    fa_sym_ld = SymbolicFlexibleArm3DOF(n_seg_mpc)
    fa_sym_hd = SymbolicFlexibleArm3DOF(n_seg)

    # Sample a random configuration
    q = pin.randomConfiguration(fa_hd.model)
    # fa.visualize(q)

    # Create initial state simulated system
    angle0_0 = 0.
    angle0_1 = 0
    angle0_2 = np.pi / 20

    q0 = np.zeros((fa_hd.nq, 1))
    q0[0] += angle0_0
    q0[1] += angle0_1
    q0[1 + n_seg + 1] += angle0_2

    dq0 = np.zeros_like(q0)
    x0 = np.vstack((q0, dq0))

    # MPC has other model states
    q0_mpc = np.zeros((fa_ld.nq, 1))
    q0_mpc[0] += angle0_0
    q0_mpc[1] += angle0_1
    q0_mpc[1 + n_seg_mpc + 1] += angle0_2

    dq0_mpc = np.zeros_like(q0_mpc)
    x0_mpc = np.vstack((q0_mpc, dq0_mpc))

    q_mpc0 = deepcopy(q0_mpc)
    dq_mpc0 = deepcopy(dq0_mpc)
    x_mpc0 = np.vstack((q_mpc0, dq_mpc0))

    _, qee0 = fa_ld.fk_ee(q0_mpc)

    # Compute reference
    q_mpc_ref = deepcopy(q_mpc0)
    q_mpc_ref[0] += np.pi / 2
    q_mpc_ref[1] += 0
    q_mpc_ref[1 + n_seg_mpc + 1] += -np.pi / 20

    dq_mpc_ref = np.zeros_like(q_mpc_ref)
    x_mpc_ref = np.vstack((q_mpc_ref, dq_mpc_ref))
    _, x_ee_ref = fa_ld.fk_ee(q_mpc_ref)

    u_ref = np.zeros((fa_sym_ld.nu, 1))  # u_ref could be changed to some known value along the trajectory

    # Create mpc options and controller
    mpc_options = Mpc3dofOptions(n_seg=n_seg_mpc)
    mpc_options.n = 30
    mpc_options.tf = 1
    mpc_options.r_diag *= 10

    if not use_NN:
        controller = Mpc3Dof(model=fa_sym_ld, x0=x_mpc0, x0_ee=qee0, options=mpc_options)
        # Choose one out of two control modes. Reference tracking uses a spline planner.
        if control_mode == ControlMode.SET_POINT:
            controller.set_reference_point(p_ee_ref=x_ee_ref, x_ref=x_mpc_ref, u_ref=u_ref)
        elif control_mode == ControlMode.REFERENCE_TRACKING:
            controller.set_reference_trajectory(q_t0=q_mpc0, pee_tf=x_ee_ref, tf=5, fun_forward_pee=fa_ld.fk_ee)
        else:
            Exception()
    else:
        controller = NNController(nn_file="bc_policy_1", n_seg=n_seg_mpc)

    # Sampling time
    dt = 0.05

    # Estimator
    est_model = SymbolicFlexibleArm3DOF(n_seg_mpc, dt=dt)
    P0 = 0.01 * np.ones((est_model.nx, est_model.nx))
    q_q, q_dq = [1e-2] * est_model.nq, [1e-1] * est_model.nq
    Q = np.diag([*q_q, *q_dq])
    r_q, r_dq, r_pee = [3e-4] * 3, [6e-2] * 3, [1e-2] * 3
    R = np.diag([*r_q, *r_dq, *r_pee])
    E = ExtendedKalmanFilter(est_model, x0_mpc, P0, Q, R)

    # simulate
    n_iter = 100
    sim = Simulator(fa_hd, controller, 'RK45', E)
    x, u, y, x_hat = sim.simulate(x0.flatten(), dt, n_iter)

    # Print timing
    if not use_NN:
        t_mean, t_std, t_min, t_max = controller.get_timing_statistics()
        print_timings(t_mean, t_std, t_min, t_max)

    # Plot result
    t = np.linspace(0, n_iter * dt, x.shape[0])
    plot_result(x=x, u=u, t=t)

    # Parse joint positions
    n_skip = 1
    q = x[::n_skip, :fa_hd.nq]

    # Animate simulated motion
    urdf_path = 'models/three_dof/three_segments/flexible_arm_3dof_3s.urdf'
    animator = Panda3dAnimator(urdf_path, dt * n_skip, q).play(30)
