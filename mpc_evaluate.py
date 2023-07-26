import numpy as np

from envs.gym_env import FlexibleArmEnvOptions, FlexibleArmEnv
from envs.flexible_arm_3dof import (SymbolicFlexibleArm3DOF,
                                get_rest_configuration,
                                inverse_kinematics_rb)
from estimator import ExtendedKalmanFilter
from mpc_3dof import Mpc3dofOptions, Mpc3Dof
from animation import Panda3dAnimator
from utils import print_timings
import plotting
import kpi


# Simulation and controller parameters
N_SEG_sim = 2
N_SEG_cntr = 2

DT = 0.01 # sampling time for contr and simulator
N_ITER = 100 # number of simulation iterations

R_Q_sim = [3e-6] * 3 # covariance of the q noise
R_DQ_sim = [2e-3] * 3 # covariance of the dq noise
R_PEE_sim = [1e-4] * 3 # covariance of the pee noise

# Estimator parameters
EST_INTEGRATOR = 'collocation'
P0_Qi = 0.01 # 0.05
P0_DQi = 1e-3
Q_Qr, Q_Qf = 1e-4, 1e-3 # r=rigid, f=flexible
Q_DQr, Q_DQf = 1e-1, 5e-1
R_Q_est = [3e-4] * 3
R_DQ_est = [5e-1] * 3
R_PEE_est = [1e-2] * 3 

# Controller parameters
N_mpc = 30
CNTR_INPUT_STATES = 'real'

# P2P task 1
# QA_t0 = np.array([np.pi / 2, np.pi / 10, -np.pi / 8])
# QA_ref = np.array([0., 2 * np.pi / 5, -np.pi / 3])

# P2P task 2
# QA_t0 = np.array([0., np.pi/20, -np.pi/20])
# QA_ref = np.array([0., np.pi/3, -np.pi/20])

# P2P task 3
QA_t0 = np.array([0., np.pi/30, -np.pi/30])
QA_ref = np.array([0., np.pi/4, -np.pi/4])

# KPI parameters
EPSILONS = [0.05]

if __name__ == "__main__":
    # Compute reference EE position
    q_ref = get_rest_configuration(QA_ref, N_SEG_sim)
    pee_ref = np.array(SymbolicFlexibleArm3DOF(N_SEG_sim).p_ee(q_ref))


    # Estimator
    est_model = SymbolicFlexibleArm3DOF(N_SEG_cntr, dt=DT, 
                                        integrator=EST_INTEGRATOR)
    P0 = np.diag([*[P0_Qi]*est_model.nq, *[P0_DQi]*est_model.nq])
    q_q = [Q_Qr, *[Q_Qf] * (est_model.nq - 1)]
    q_dq = [Q_DQr, *[Q_DQf] * (est_model.nq - 1)] 
    Q = np.diag([*q_q, *q_dq])
    R = np.diag([*R_Q_est, *R_DQ_est, *R_PEE_est])

    qa0_est = QA_t0.copy()
    q0_est = get_rest_configuration(qa0_est, N_SEG_cntr)
    dq0_est = np.zeros_like(q0_est)
    x0_est = np.vstack((q0_est, dq0_est))

    E = ExtendedKalmanFilter(est_model, x0_est, P0, Q, R)


    # gym environment
    env_opts = FlexibleArmEnvOptions(
        dt = DT,
        qa_start = QA_t0,
        qa_end = QA_ref,
        qa_range_start = np.zeros(3),
        n_seg = N_SEG_sim,
        sim_time = N_ITER*DT,
        sim_noise_R = np.diag([*R_Q_sim, *R_DQ_sim, *R_PEE_sim]),
        contr_input_states = CNTR_INPUT_STATES
    )
    env = FlexibleArmEnv(env_opts, E)


    # Controller
    cntr_model = SymbolicFlexibleArm3DOF(N_SEG_cntr)
    mpc_opts = Mpc3dofOptions(
        n_seg = N_SEG_cntr,
        tf = N_mpc*DT,
        n = N_mpc,
    )
    x0_mpc = x0_est.copy()
    pee_0 = np.array(cntr_model.p_ee(q0_est))

    C = Mpc3Dof(model=cntr_model, x0=x0_mpc, pee_0=pee_0, options=mpc_opts)
    assert mpc_opts.get_sampling_time() == DT

    # Provide reference for the mpc controller
    # get qa from IK of the rigid body approximation
    qa_ik = inverse_kinematics_rb(pee_ref, QA_t0)
    q_ref_mpc = get_rest_configuration(qa_ik, N_SEG_cntr)
    dq_ref_mpc = np.zeros_like(q_ref_mpc)
    x_ref_mpc = np.vstack((q_ref_mpc, dq_ref_mpc))
    # reference input (can be initialzed using RNEA for rigid body model)
    u_ref = cntr_model.gravity_torque(q_ref_mpc)
    C.set_reference_point(p_ee_ref=pee_ref, x_ref=x_ref_mpc, u_ref=u_ref)


    # Simulate with environment
    nq = cntr_model.nq
    state = env.reset()
    qk, dqk = np.expand_dims(state[0:nq], 1), np.expand_dims(state[nq:], 1)

    for i in range(env.max_intg_steps):
        a = C.compute_torques(q=qk, dq=dqk, t=i*DT)
        state, reward, done, info = env.step(a)
        qk, dqk = np.expand_dims(state[0:nq], 1), np.expand_dims(state[nq:], 1)
        if done:
            break


    # Parse simulation results
    x, u, y, xhat = env.simulator.x, env.simulator.u, env.simulator.y, env.simulator.x_hat
    t = np.arange(0, env.simulator.opts.n_iter + 1) * env.dt

    # Compute KPIs
    t_mean, t_std, t_min, t_max = C.get_timing_statistics()
    print_timings(t_mean, t_std, t_min, t_max)

    q = x[:, :env.model_sym.nq]
    t_epsilons = []
    pls = []
    for epsilon in EPSILONS:
        ns2g = kpi.execution_time(q, env.simulator.robot, env.xee_final, epsilon)
        pl = -1
        if ns2g >= 0:
            q_kpi = q[:ns2g, :]
            pl = kpi.path_length(q_kpi, env.simulator.robot)
        t_epsilons.append(ns2g * env.dt)
        pls.append(pl)
        print(f"Path length = {pl:.4f}")
        print(f"Time to goal = {(ns2g * env.dt):.4f}")


    # Visualization
    plotting.plot_real_states_vs_estimate(t, x, xhat, N_SEG_sim, N_SEG_cntr)
    plotting.plot_measurements(t, y, pee_ref)

    Panda3dAnimator(env.model_sym.urdf_path, env.dt, q).play(2)