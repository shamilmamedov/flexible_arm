"""
This demo uses MPC as the expert collects data for the FlexibleArmEnv environment.
RUN COMMAND: python -m tests.test_mpc_data_collection
"""

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data import serialize

from estimator import ExtendedKalmanFilter
from envs.flexible_arm_3dof import FlexibleArm3DOF, get_rest_configuration
from envs.gym_env import FlexibleArmEnv, FlexibleArmEnvOptions, SymbolicFlexibleArm3DOF
from mpc_3dof import Mpc3dofOptions, Mpc3Dof
from utils.gym_utils import CallableExpert


rng = np.random.default_rng(0)

# --- Create FlexibleArm environment ---
# symbolic model (think of it as the computational graph)
# normal model (think of it as the actual simulation model)
# --- data env ---
n_seg_data = 3
# fa_data = FlexibleArm3DOF(n_seg_data)
fa_sym_data = SymbolicFlexibleArm3DOF(n_seg_data)

# --- control env ---
n_seg_control = 3
fa_control = FlexibleArm3DOF(n_seg_control)
fa_sym_control = SymbolicFlexibleArm3DOF(n_seg_control)


# Create initial state data env
qa0 = np.array(
    [np.pi / 2, np.pi / 10, -np.pi / 8]
)  # base-rotation, base-bend, elbow-bend
q0 = get_rest_configuration(qa0, n_seg_data)
dq0 = np.zeros_like(q0)
x0 = np.vstack((q0, dq0))  # state: active and passive joint angles and velocities
# _, xee0 = fa_data.fk_ee(
#     q0
# )  # end-effector position in cartesian space (ignore the first output for now)

# Compute reference data env
# qa_ref = np.array([0.0, 2 * np.pi / 5, -np.pi / 3])
# q_ref = get_rest_configuration(qa_ref, n_seg)
# dq_ref = np.zeros_like(q_ref)
# x_ref = np.vstack((q_ref, dq_ref))
# _, x_ee_ref = fa_ld.fk_ee(q_ref)

# create initial state control env
qa0_control = np.array(
    [np.pi / 2, np.pi / 10, -np.pi / 8]
)  # base-rotation, base-bend, elbow-bend
q0_control = get_rest_configuration(qa0_control, n_seg_control)
dq0_control = np.zeros_like(q0_control)
x0_control = np.vstack(
    (q0_control, dq0_control)
)  # state: active and passive joint angles and velocities
_, xee0_control = fa_control.fk_ee(
    q0_control
)  # end-effector position in cartesian space (ignore the first output for now)

# Compute reference control env (Workflow should be: EE -> JOINT STATES but we do it the other way around for convinience)
qa_ref_control = np.array([0.0, 2 * np.pi / 5, -np.pi / 3])
q_ref_control = get_rest_configuration(qa_ref_control, n_seg_control)
dq_ref_control = np.zeros_like(q_ref_control)
x_ref_control = np.vstack((q_ref_control, dq_ref_control))
_, x_ee_ref_control = fa_control.fk_ee(
    q_ref_control
)  # GOAL SPECIFICATION IN CARTESIAN SPACE


def get_active_joint_angles(q, n_seg):
    """
    get elements at 0, 1, and 2 + n_seg
    """
    return np.vstack((q[0:2], q[2 + n_seg])).flatten()


# --- Create Extended Kalman Filter ---
fa_sym_estimator = SymbolicFlexibleArm3DOF(n_seg_control, integrator="cvodes")
p0_q, p0_dq = [0.05] * fa_sym_estimator.nq, [1e-3] * fa_sym_estimator.nq
P0 = np.diag([*p0_q, *p0_dq])
q_q = [1e-4, *[1e-3] * (fa_sym_estimator.nq - 1)]
q_dq = [1e-1, *[5e-1] * (fa_sym_estimator.nq - 1)]
Q = np.diag([*q_q, *q_dq])
r_q, r_dq, r_pee = [3e-5] * 3, [5e-2] * 3, [1e-3] * 3
R = 10 * np.diag([*r_q, *r_dq, *r_pee])
estimator = ExtendedKalmanFilter(fa_sym_estimator, x0_control, P0, Q, R)

# create data environment
# Measurement noise covairance parameters
# R_Q = [3e-6] * 3
# R_DQ = [2e-3] * 3
# R_PEE = [1e-4] * 3
zero_noise_cov = np.zeros((9, 9))
env_options = FlexibleArmEnvOptions(
    n_seg=n_seg_data,
    dt=0.01,
    qa_start=get_active_joint_angles(q0, n_seg_data),
    qa_end=get_active_joint_angles(q_ref_control, n_seg_control),
    contr_input_states="real",  # "real" if the n_seg is the same for the data and control env
    sim_noise_R=zero_noise_cov,  # np.diag([*R_Q, *R_DQ, *R_PEE]),
    render_mode="human",
)
env = FlexibleArmEnv(env_options, estimator=None)
# --------------------------------------

# --- Create MPC controller ---

# create expert MPC
mpc_options = Mpc3dofOptions(n_seg=n_seg_control)
mpc_options.n = 130
mpc_options.tf = 1.3
# mpc_options.r_diag *= 10
# mpc_options.nlp_iter = 100
controller = Mpc3Dof(
    model=fa_sym_control, x0=x0_control, pee_0=xee0_control, options=mpc_options
)
u_ref = np.zeros(
    (fa_sym_control.nu, 1)
)  # u_ref could be changed to some known value along the trajectory

# Choose one out of two control modes. Reference tracking uses a spline planner.
controller.set_reference_point(
    p_ee_ref=x_ee_ref_control, x_ref=x_ref_control, u_ref=u_ref
)

# create MPC expert
expert = CallableExpert(
    controller,
    observation_space=env.observation_space,
    action_space=env.action_space,
)
# -----------------------------

# --- Collect expert trajectories ---
print("Sampling expert transitions.")
rollouts = rollout.rollout(
    expert,
    DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
    rollout.make_sample_until(min_timesteps=None, min_episodes=10),
    rng=rng,
    verbose=True,
)
serialize.save("mpc_expert_rollouts.pkl", rollouts)
# -----------------------------------
