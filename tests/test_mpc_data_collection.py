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


env, expert, _ = create_unified_flexiblearmenv_and_controller_and_safety_filter(
    create_controller=True, add_wall_obstacle=False, create_safety_filter=False
)

# --- Collect expert trajectories ---
print("Sampling expert transitions.")
rollouts = rollout.rollout(
    expert,
    DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
    rollout.make_sample_until(min_timesteps=None, min_episodes=100),
    rng=rng,
    verbose=True,
    render=False,
)
serialize.save("mpc_expert_rollouts.pkl", rollouts)
# -----------------------------------
