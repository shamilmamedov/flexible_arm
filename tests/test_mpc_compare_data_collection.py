"""
This demo uses MPC as the expert collects data for the FlexibleArmEnv environment.
RUN COMMAND: python -m tests.test_mpc_data_collection
"""

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data import serialize
from utils.gym_utils import (
    create_unified_flexiblearmenv_and_controller_and_safety_filter,
)

SEED = 0
rng = np.random.default_rng(SEED)
USE_OBSTACLE = False
N_EPS = 10

n_seg_mpc = 3 # this should be removed here!
#env_options = {"flex_param_file_path": "/home/rudolf/PycharmProjects/flexible_arm/models/three_dof/one_segments/flexibility_params.yml"}
env, expert, _ = create_unified_flexiblearmenv_and_controller_and_safety_filter(
    controller_type="mpc_phases",
    create_controller=True,
    add_wall_obstacle=USE_OBSTACLE,
    create_safety_filter=False,
    n_seg_mpc=n_seg_mpc,
 #   env_opts=env_options
)

# --- Collect expert trajectories ---
print("Sampling phases transitions.")
rollouts = rollout.rollout(
    expert,
    DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
    rollout.make_sample_until(min_timesteps=None, min_episodes=N_EPS),
    rng=rng,
    verbose=True,
    render=False,
)
serialize.save("mpc_expert_phases_rollouts_"+str(n_seg_mpc)+".pkl", rollouts)
print(f"Timings phases: mean={np.mean(expert.controller.debug_timings)}, std: {np.std(expert.controller.debug_timings)}")

n_seg_mpc = 1
#env_options = {"flex_param_file_path": "/home/rudolf/PycharmProjects/flexible_arm/models/three_dof/one_segments/flexibility_params.yml"}
env, expert, _ = create_unified_flexiblearmenv_and_controller_and_safety_filter(
    controller_type="mpc",
    create_controller=True,
    add_wall_obstacle=USE_OBSTACLE,
    create_safety_filter=False,
    n_seg_mpc=n_seg_mpc,
 #   env_opts=env_options
)

# --- Collect expert trajectories ---
print("Sampling mpc with 1 segment transitions.")
rollouts = rollout.rollout(
    expert,
    DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
    rollout.make_sample_until(min_timesteps=None, min_episodes=N_EPS),
    rng=rng,
    verbose=True,
    render=False,
)
serialize.save("mpc_expert_rollouts_"+str(n_seg_mpc)+".pkl", rollouts)
print(f"Timings: mean={np.mean(expert.controller.debug_timings)}, std: {np.std(expert.controller.debug_timings)}")



# -----------------------------------
n_seg_mpc = 3
env, expert, _ = create_unified_flexiblearmenv_and_controller_and_safety_filter(
    create_controller=True,
    add_wall_obstacle=USE_OBSTACLE,
    create_safety_filter=False,
    n_seg_mpc=n_seg_mpc
)

# --- Collect expert trajectories ---
print("Sampling mps with 3 segments transitions.")
rollouts = rollout.rollout(
    expert,
    DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
    rollout.make_sample_until(min_timesteps=None, min_episodes=N_EPS),
    rng=rng,
    verbose=True,
    render=False,
)
serialize.save("mpc_expert_rollouts_"+str(n_seg_mpc)+".pkl", rollouts)
print(f"Timings: mean={np.mean(expert.controller.debug_timings)}, std: {np.std(expert.controller.debug_timings)}")
# -----------------------------------
n_seg_mpc = 5
env, expert, _ = create_unified_flexiblearmenv_and_controller_and_safety_filter(
    create_controller=True,
    add_wall_obstacle=USE_OBSTACLE,
    create_safety_filter=False,
    n_seg_mpc=n_seg_mpc
)

# --- Collect expert trajectories ---
print("Sampling mpc with 5 segments transitions.")
rollouts = rollout.rollout(
    expert,
    DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
    rollout.make_sample_until(min_timesteps=None, min_episodes=N_EPS),
    rng=rng,
    verbose=True,
    render=False,
)
serialize.save("mpc_expert_rollouts_"+str(n_seg_mpc)+".pkl", rollouts)
print(f"Timings: mean={np.mean(expert.controller.debug_timings)}, std: {np.std(expert.controller.debug_timings)}")