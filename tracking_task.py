import numpy as np

from utils import print_timings
from flexible_arm_3dof import (SymbolicFlexibleArm3DOF, inverse_kinematics_rb,
                               get_rest_configuration)
from estimator import ExtendedKalmanFilter
from mpc_3dof import Mpc3dofOptions, Mpc3Dof
from controller import NNController
from simulation import SimulatorOptions, Simulator
from animation import Panda3dAnimator
import plotting
import kpi


if __name__ == "__main__":
    # Model discretization parameters
    n_seg_sim = 10
    n_seg_cont = 2

    # Sampling time for both controller and simulator
    dt = 0.01

    # Number of simulation iterations
    n_iter = 150

    # Controller choice
    controllers = ['MPC', 'NN']
    cont_name = controllers[0]

    # Trajectory parameters: the end-effector position
    # that specifies the center of the circle
    T_traj = 0.75
    R_circle = 0.2
    qa_t0_traj = np.array([0., 2*np.pi/5, -np.pi/3])


    # Create model instances
    sim_model = SymbolicFlexibleArm3DOF(n_seg_sim)
    cont_model = SymbolicFlexibleArm3DOF(n_seg_cont)
    est_model = SymbolicFlexibleArm3DOF(n_seg_cont, dt=dt, integrator='collocation')

    # Initial state of the real system (simulation model)
    qa0 = np.array([0., 2*np.pi/5, -np.pi/3])
    q0 = get_rest_configuration(qa0, n_seg_sim)
    dq0 = np.zeros_like(q0)
    x0 = np.vstack((q0, dq0))

    
    # Design estimator
    # initial covariance matrix
    p0_q = [0.05] * est_model.nq  # 0.05, 0.1
    p0_dq = [1e-3] * est_model.nq
    P0 = np.diag([*p0_q, *p0_dq])
    # process noise covariance
    q_q = [1e-4, *[1e-3] * (est_model.nq - 1)]
    q_dq = [1e-1, *[5e-1] * (est_model.nq - 1)]  # 5e-1, 10e-1
    Q = np.diag([*q_q, *q_dq])
    # measurement noise covaiance
    r_q, r_dq, r_pee = [3e-5] * 3, [5e-2] * 3, [1e-3] * 3
    R = 10 * np.diag([*r_q, *r_dq, *r_pee])

    # initial state for the estimator
    qa0_est = qa0.copy()
    q0_est = get_rest_configuration(qa0_est, n_seg_cont)
    dq0_est = np.zeros_like(q0_est)
    x0_est = np.vstack((q0_est, dq0_est))

    E = ExtendedKalmanFilter(est_model, x0_est, P0, Q, R)


    # Design controller
    if cont_name == 'MPC':
        # Initial state of the MPC controller
        # which initial state guess of the state estimator
        x0_mpc = x0_est.copy()

        # Initial value of the end-effector
        pee_0 = np.array(cont_model.p_ee(q0_est))

        # Instantiate an MPC conttroller
        mpc_options = Mpc3dofOptions(n_seg=n_seg_cont, tf=0.3)
        controller = Mpc3Dof(model=cont_model, x0=x0_mpc, pee_0=pee_0, options=mpc_options)
        assert mpc_options.get_sampling_time() == dt


        pee_iterp_fcn = controller.set_reference_trajectory(
                            qa_t0=qa_t0_traj, tf=T_traj, r=R_circle)
    elif cont_name == 'NN':
        controller = NNController(nn_file="bc_policy_1", n_seg=n_seg_cont)
    else:
        raise NotImplementedError


    # Simulate the robot
    sim_opts = SimulatorOptions(contr_input_states='estimated')
    sim_opts.dt = dt
    sim = Simulator(sim_model, controller, 'cvodes', E, opts=sim_opts)
    x, u, y, xhat = sim.simulate(x0.flatten(), n_iter)
    t = np.arange(0, n_iter + 1) * dt

    if cont_name == 'MPC':
        # Print timing
        t_mean, t_std, t_min, t_max = controller.get_timing_statistics()
        print_timings(t_mean, t_std, t_min, t_max)

    # Compute KPIs
    q = x[:, :sim_model.nq]
    dq = x[:, sim_model.nq:]
    u_viol, dqa_viol = kpi.constraint_violation(q, dq, u, sim_model)
    print(f"Control constraint violation: {u_viol.tolist()}")
    print(f"Velocity constraint violation: {dqa_viol.tolist()}")

    # Process the simulation results
    # Parse joint positions
    n_skip = 1
    q = x[::n_skip, :sim_model.nq]

    # Visualization
    # plotting.plot_real_states_vs_estimate(t, x, xhat, n_seg_sim, n_seg_cont)
    # plotting.plot_controls(t[:-1], u, u_ref)
    plotting.plot_measurements(t, y, pee_iterp_fcn(t))

    # Animate simulated motion
    animator = Panda3dAnimator(sim_model.urdf_path, dt * n_skip, q).play(2)
