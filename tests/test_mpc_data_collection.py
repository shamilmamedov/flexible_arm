"""
This demo uses MPC as the expert collects data for the FlexibleArmEnv environment.
RUN COMMAND: python -m tests.test_mpc_data_collection
"""

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data import serialize


from envs.flexible_arm_3dof import SymbolicFlexibleArm3DOF, get_rest_configuration
from envs.gym_env import (
    FlexibleArmEnv,
    FlexibleArmEnvOptions,
)
from mpc_3dof import Mpc3dofOptions, Mpc3Dof
from utils.gym_utils import CallableMPCExpert
from utils.utils import StateType


rng = np.random.default_rng(0)


# --- Create FlexibleArm environment ---
n_seg = 5
n_seg_mpc = 3

# Create initial/final state: (base-rotation, base-bend, elbow-bend)
qa_initial = np.array([np.pi / 2, np.pi / 10, -np.pi / 8])
qa_final = np.array([0.0, 2 * np.pi / 5, -np.pi / 3])

# create data environment
R_Q = [3e-6] * 3
R_DQ = [2e-3] * 3
R_PEE = [1e-4] * 3
env_options = FlexibleArmEnvOptions(
    n_seg=n_seg,
    n_seg_estimator=n_seg_mpc,
    sim_time=1.3,
    dt=0.01,
    qa_start=qa_initial,
    qa_end=qa_final,
    qa_range_end=np.array([1.0, 1.0, 1.0]),
    contr_input_states=StateType.ESTIMATED,  # "real" if the n_seg is the same for the data and control env
    sim_noise_R=np.diag([*R_Q, *R_DQ, *R_PEE]),
    render_mode="human",
)
env = FlexibleArmEnv(env_options)
# env.reset()
# for _ in range(500):
#     env.render()
# env.renderer.viz.viewer.stop()
# --------------------------------------

# --- Create MPC controller ---
fa_sym_mpc = SymbolicFlexibleArm3DOF(n_seg_mpc)
mpc_options = Mpc3dofOptions(n_seg=n_seg_mpc, tf=1.3, n=130)
controller = Mpc3Dof(model=fa_sym_mpc, x0=None, pee_0=None, options=mpc_options)


# create MPC expert
expert = CallableMPCExpert(
    controller,
    observation_space=env.observation_space,
    action_space=env.action_space,
    observation_includes_goal=True,
)
# -----------------------------

# --- Collect expert trajectories ---
print("Sampling expert transitions.")
rollouts = rollout.rollout(
    expert,
    DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
    rollout.make_sample_until(min_timesteps=None, min_episodes=100),
    rng=rng,
    verbose=True,
)
serialize.save("mpc_expert_rollouts.pkl", rollouts)
# -----------------------------------
