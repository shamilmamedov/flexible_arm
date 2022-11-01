import numpy as np


from utils import print_timings
from flexible_arm_3dof import (SymbolicFlexibleArm3DOF,
                               get_rest_configuration)
from estimator import ExtendedKalmanFilter
from mpc_3dof import Mpc3dofOptions, Mpc3Dof
from controller import NNController, PDController3Dof
from simulation import SimulatorOptions, Simulator
from animation import Panda3dAnimator
import plotting


if __name__ == "__main__":
    # Model discretization parameters
    n_seg_sim = 10
    n_seg_cont = 3

    # Sampling time for both controller and simulator
    dt = 0.01 

    # Number of simulation iterations
    n_iter = 200

    # Controller
    controllers = ['MPC', 'NN']
    controller = controllers[0]

    # Create model instances
    sim_model = SymbolicFlexibleArm3DOF(n_seg_sim)
    cont_model = SymbolicFlexibleArm3DOF(n_seg_cont)
    est_model = SymbolicFlexibleArm3DOF(n_seg_cont, dt=dt, integrator='cvodes')

    # Initial state of the real system (simulation model)
    qa0 = np.array([0.1, 1.5, 0.5])
    q0 = get_rest_configuration(qa0, n_seg_sim)
    dq0 = np.zeros_like(q0)
    x0 = np.vstack((q0, dq0))

    # Compute reference
    qa_ref = qa0.copy()
    qa_ref += np.array([1.5, 0.5, 1.5])
    q_ref = get_rest_configuration(qa_ref, n_seg_sim)
    pee_ref = np.array(sim_model.p_ee(q_ref))

    # Design estimator
    # initial covariance matrix
    p0_q, p0_dq = [0.05] * est_model.nq, [1e-3] * est_model.nq
    P0 = np.diag([*p0_q, *p0_dq])
    # process noise covariance
    q_q, q_dq = [1e-2] * est_model.nq, [5e-1] * est_model.nq
    Q = np.diag([*q_q, *q_dq])
    # measurement noise covaiance
    r_q, r_dq, r_pee = [3e-5] * 3, [5e-2] * 3, [1e-3] * 3
    R = np.diag([*r_q, *r_dq, *r_pee])

    # initial state for the estimator
    qa0_est = qa0.copy()
    q0_est = get_rest_configuration(qa0_est, n_seg_cont)
    dq0_est = np.zeros_like(q0_est)
    x0_est = np.vstack((q0_est, dq0_est))

    E = ExtendedKalmanFilter(est_model, x0_est, P0, Q, R)

    if controller == 'MPC':
        # Initial state of the MPC controller
        # which initial state guess of the state estimator
        x0_mpc = x0_est.copy()

        # Initial value of the end-effector
        pee_0 = np.array(cont_model.p_ee(q0_est))

        mpc_options = Mpc3dofOptions(n_seg=n_seg_cont, tf=1)
        controller = Mpc3Dof(model=cont_model, x0=x0_mpc, pee_0=pee_0, options=mpc_options)
        
        # reference input (can be initialzed using RNEA for rigid body model)
        u_ref = np.zeros((cont_model.nu, 1))

        # reference for the mpc controller
        controller.set_reference_point(p_ee_ref=pee_ref, x_ref=x_mpc_ref, u_ref=u_ref)
    elif controller == 'NN':
        controller = NNController(nn_file="bc_policy_1", n_seg=n_seg_cont)


    # Simulate the robot
    sim_opts = SimulatorOptions(contr_input_states='estimated')
    sim = Simulator(sim_model, controller, 'cvodes', E, opts=sim_opts)
    x, u, y, xhat = sim.simulate(x0.flatten(), dt, n_iter)
    t = np.arange(0, n_iter + 1) * dt

    if controller == 'MPC':
        # Print timing
        t_mean, t_std, t_min, t_max = controller.get_timing_statistics()
        print_timings(t_mean, t_std, t_min, t_max)


    # Parse joint positions
    n_skip = 1
    q = x[::n_skip, :sim_model.nq]

    # Visualization
    plotting.plot_controls(t[:-1], u)
    plotting.plot_measurements(t, y)

    # Animate simulated motion
    animator = Panda3dAnimator(sim_model.urdf_path, dt*n_skip, q).play(3)

