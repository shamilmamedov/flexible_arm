"""
This demo uses MPC as the expert collects data for the FlexibleArmEnv environment.
RUN COMMAND: python -m tests.test_mpc_data_collection
"""

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data import serialize

from envs.flexible_arm_3dof import FlexibleArm3DOF, get_rest_configuration
from envs.gym_env import (
    FlexibleArmEnv,
    FlexibleArmEnvOptions,
    SymbolicFlexibleArm3DOF,
)
from mpc_3dof import Mpc3dofOptions, Mpc3Dof
from utils.gym_utils import CallableExpert
from utils.utils import StateType


rng = np.random.default_rng(0)


# --- Create FlexibleArm environment ---
# symbolic model (think of it as the computational graph)
# normal model (think of it as the actual simulation model)
# --- data env ---
n_seg_data = 3

# --- control env ---
n_seg_control = 3
fa_control = FlexibleArm3DOF(n_seg_control)
fa_sym_control = SymbolicFlexibleArm3DOF(n_seg_control)


# Create initial state data env
qa0 = np.array(
    [np.pi / 2, np.pi / 10, -np.pi / 8]
)  # base-rotation, base-bend, elbow-bend
q0 = get_rest_configuration(qa0, n_seg_data)


# create initial state control env
qa0_control = qa0.copy()
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


# create data environment
R_Q = [3e-6] * 3
R_DQ = [2e-3] * 3
R_PEE = [1e-4] * 3
env_options = FlexibleArmEnvOptions(
    n_seg=n_seg_data,
    n_seg_estimator=n_seg_control,
    sim_time=1.3,
    dt=0.01,
    qa_start=qa0,
    qa_end=qa_ref_control,
    contr_input_states=StateType.ESTIMATED,  # "real" if the n_seg is the same for the data and control env
    sim_noise_R=np.diag([*R_Q, *R_DQ, *R_PEE]),
    render_mode="human",
)
env = FlexibleArmEnv(env_options)
# --------------------------------------

# --- Create MPC controller ---

# create expert MPC
mpc_options = Mpc3dofOptions(n_seg=n_seg_control, tf=1.3)
mpc_options.n = 130
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

print(f"MPC controller goal:{x_ee_ref_control}")
env.reset()
print(f"env_goal:{env.xee_final}")

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
