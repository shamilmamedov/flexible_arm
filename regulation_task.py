import numpy as np

from utils import print_timings
from envs.flexible_arm_3dof import (SymbolicFlexibleArm3DOF, inverse_kinematics_rb,
                               get_rest_configuration)
from estimator import ExtendedKalmanFilter
from mpc_3dof import Mpc3dofOptions, Mpc3Dof
from controller import NNController, ConstantController
from simulation import SimulatorOptions, Simulator
from animation import Panda3dAnimator
import plotting
import kpi


N_SEG_sim = 10
N_SEG_cntr = 2

DT = 0.01 # sampling time for contr and simulator
N_ITER = 100 # number of simulation iterations

# P2P task 1
QA_t0 = np.array([np.pi / 2, np.pi / 10, -np.pi / 8])
QA_ref = np.array([0., 2 * np.pi / 5, -np.pi / 3])

# P2P task 2
# QA_t0 = np.array([0., np.pi/20, -np.pi/20])
# QA_ref = np.array([0., np.pi/3, -np.pi/20])

# P2P task 3
# QA_t0 = np.array([0., np.pi/30, -np.pi/30])
# QA_ref = np.array([0., np.pi/4, -np.pi/4])


if __name__ == "__main__":

    # Controller
    controllers = ['MPC', 'NN', 'PD']
    cont_name = controllers[0]

    # Create model instances
    sim_model = SymbolicFlexibleArm3DOF(N_SEG_sim)
    cont_model = SymbolicFlexibleArm3DOF(N_SEG_cntr)
    est_model = SymbolicFlexibleArm3DOF(N_SEG_cntr, dt=DT, integrator='collocation')

    # Initial state of the real system (simulation model)
    q0 = get_rest_configuration(QA_t0, N_SEG_sim)
    dq0 = np.zeros_like(q0)
    x0 = np.vstack((q0, dq0))

    # Compute reference ee position
    q_ref = get_rest_configuration(QA_ref, N_SEG_sim)
    pee_ref = np.array(sim_model.p_ee(q_ref))
    print("ref: " + np.array2string(QA_ref.squeeze()))

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
    qa0_est = QA_t0.copy()
    q0_est = get_rest_configuration(qa0_est, N_SEG_cntr)
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
        mpc_options = Mpc3dofOptions(n_seg=N_SEG_cntr, tf=0.3)
        controller = Mpc3Dof(model=cont_model, x0=x0_mpc, pee_0=pee_0, options=mpc_options)
        assert mpc_options.get_sampling_time() == DT

        # Provide reference for the mpc controller
        # get qa from IK of the rigid body approximation
        qa_ik = inverse_kinematics_rb(pee_ref, QA_t0)
        q_ref_mpc = get_rest_configuration(qa_ik, N_SEG_cntr)
        dq_ref_mpc = np.zeros_like(q_ref_mpc)
        x_ref_mpc = np.vstack((q_ref_mpc, dq_ref_mpc))
        # reference input (can be initialzed using RNEA for rigid body model)
        u_ref = cont_model.gravity_torque(q_ref_mpc)
        controller.set_reference_point(p_ee_ref=pee_ref, x_ref=x_ref_mpc, u_ref=u_ref)
    elif cont_name == 'NN':
        controller = NNController(nn_file="bc_policy_1", n_seg=N_SEG_cntr)
    elif cont_name == 'PD':
        tau_const = cont_model.gravity_torque(q0_est)
        controller = ConstantController(tau_const)  
    else:
        raise NotImplementedError


    # Simulate the robot
    sim_opts = SimulatorOptions(
        dt = DT,
        # R = np.zeros((9, 9)),
        contr_input_states='estimated')
    sim = Simulator(sim_model, controller, 'cvodes', E, opts=sim_opts)
    x, u, y, xhat = sim.simulate(x0.flatten(), N_ITER)
    t = np.arange(0, N_ITER + 1) * DT

    if cont_name == 'MPC':
        # Print timing
        t_mean, t_std, t_min, t_max = controller.get_timing_statistics()
        print_timings(t_mean, t_std, t_min, t_max)

    # Compute KPIs
    q = x[:, :sim_model.nq]
    dq = x[:, sim_model.nq:]
    ns2g = kpi.execution_time(q, sim_model, pee_ref, 0.05)
    print(f"Time to reach a ball around the goal = {t[ns2g]}")

    q_kpi = q[:ns2g, :]
    pl = kpi.path_length(q_kpi, sim_model)
    print(f"Path length = {pl:.4f}")

    kpi.constraint_violation(q, dq, u, sim_model)

    # Process the simulation results
    # Parse joint positions
    n_skip = 1
    q = x[::n_skip, :sim_model.nq]

    # Visualization
    # plotting.plot_real_states_vs_estimate(t, x, xhat, N_SEG_sim, N_SEG_cntr)
    # plotting.plot_controls(t[:-1], u, cont_model.gravity_torque(q0_est)) # u_ref
    plotting.plot_measurements(t, y, pee_ref, dqa_max=cont_model.dqa_max)

    # Animate simulated motion
    animator = Panda3dAnimator(sim_model.urdf_path, DT * n_skip, q).play(2)
