from copy import deepcopy
import numpy as np
from estimator import ExtendedKalmanFilter
from utils.utils import plot_result, print_timings, ControlMode
from envs.flexible_arm_3dof import (FlexibleArm3DOF, SymbolicFlexibleArm3DOF,
                               get_rest_configuration)
from animation import Panda3dAnimator
from mpc_3dof import Mpc3dofOptions, Mpc3Dof
from simulation import Simulator, SimulatorOptions
import plotting

if __name__ == "__main__":
    # options
    control_mode = ControlMode.SET_POINT

    # Create FlexibleArm instances
    n_seg = 10
    n_seg_mpc = 2

    fa_ld = FlexibleArm3DOF(n_seg_mpc)
    fa_hd = FlexibleArm3DOF(n_seg)
    fa_sym_ld = SymbolicFlexibleArm3DOF(n_seg_mpc)
    fa_sym_hd = SymbolicFlexibleArm3DOF(n_seg)

    # Create initial state simulated system
    qa0 = np.array([np.pi/2, np.pi/10, -np.pi/8])
    q0 = get_rest_configuration(qa0, n_seg)
    dq0 = np.zeros_like(q0)
    x0 = np.vstack((q0, dq0))

    # MPC has other model states
    qa0_mpc = qa0.copy()
    q0_mpc = get_rest_configuration(qa0_mpc, n_seg_mpc)
    dq0_mpc = np.zeros_like(q0_mpc)
    x0_mpc = np.vstack((q0_mpc, dq0_mpc))

    q_mpc0 = deepcopy(q0_mpc)
    dq_mpc0 = deepcopy(dq0_mpc)
    x_mpc0 = np.vstack((q_mpc0, dq_mpc0))

    _, qee0 = fa_ld.fk_ee(q0_mpc)

    # Compute reference
    # qa_mpc_ref = deepcopy(qa0_mpc)
    qa_mpc_ref = np.array([0., 2*np.pi/5, -np.pi/3])
    q_mpc_ref = get_rest_configuration(qa_mpc_ref, n_seg_mpc)
    dq_mpc_ref = np.zeros_like(q_mpc_ref)
    x_mpc_ref = np.vstack((q_mpc_ref, dq_mpc_ref))
    _, x_ee_ref = fa_ld.fk_ee(q_mpc_ref)

    # Create mpc options and controller
    mpc_options = Mpc3dofOptions(n_seg=n_seg_mpc, tf=1.3)
    mpc_options.n = 130
    controller = Mpc3Dof(model=fa_sym_ld, x0=x_mpc0, pee_0=qee0, options=mpc_options)
    u_ref = np.zeros((fa_sym_ld.nu, 1))  # u_ref could be changed to some known value along the trajectory

    # Choose one out of two control modes. Reference tracking uses a spline planner.
    if control_mode == ControlMode.SET_POINT:
        controller.set_reference_point(p_ee_ref=x_ee_ref, x_ref=x_mpc_ref, u_ref=u_ref)
    elif control_mode == ControlMode.REFERENCE_TRACKING:
        controller.set_reference_trajectory(q_t0=q_mpc0, pee_tf=x_ee_ref, 
                                            tf=5, fun_forward_pee=fa_ld.fk_ee)
    else:
        raise ValueError

    # Sampling time
    dt = 0.01

    # Estimator
    est_model = SymbolicFlexibleArm3DOF(n_seg_mpc, dt=dt, integrator='cvodes')
    p0_q, p0_dq = [0.05] * est_model.nq, [1e-3] * est_model.nq
    P0 = np.diag([*p0_q, *p0_dq])
    q_q = [1e-4, *[1e-3] * (est_model.nq - 1)]
    q_dq = [1e-1, *[5e-1] * (est_model.nq - 1)]
    Q = np.diag([*q_q, *q_dq])
    r_q, r_dq, r_pee = [3e-5] * 3, [5e-2] * 3, [1e-3] * 3
    R = 10*np.diag([*r_q, *r_dq, *r_pee])
    E = ExtendedKalmanFilter(est_model, x0_mpc, P0, Q, R)

    #controller = NNController(nn_file="trained_policy", n_seg=3)

    # simulate
    n_iter = 100
    sim_opts = SimulatorOptions(contr_input_states='estimated', dt=dt)
    sim = Simulator(fa_sym_hd, controller, 'cvodes', E, opts=sim_opts)
    x, u, y, xhat = sim.simulate(x0.flatten(), n_iter)
    t = np.arange(0, n_iter + 1) * dt

    # Print timing
    t_mean, t_std, t_min, t_max = controller.get_timing_statistics()
    print_timings(t_mean, t_std, t_min, t_max)

    # Parse joint positions
    n_skip = 1
    q = x[::n_skip, :fa_sym_hd.nq]

    # Visualization
    # plotting.plot_controls(t[:-1], u)
    plotting.plot_measurements(t, y, x_ee_ref)

    # Animate simulated motion
    animator = Panda3dAnimator(fa_sym_hd.urdf_path, dt*n_skip, q).play(3)
