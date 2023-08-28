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
    create_unified_flexiblearmenv_and_controller_and_safety_filter, SafetyWrapper,
)

SEED = 0
rng = np.random.default_rng(SEED)

# set goal behind wall
env_opts = {"qa_goal": np.array([-np.pi / 2, 0, 0])}

# turn of obstacle avoidance within MPC
cntrl_opts = {"wall_constraint_on": False}

# prepare elements for rl environment
env, expert, safety_filter = create_unified_flexiblearmenv_and_controller_and_safety_filter(
    create_controller=True, add_wall_obstacle=True, create_safety_filter=True,
    env_opts=env_opts, cntrl_opts=cntrl_opts
)
expert = SafetyWrapper(expert, safety_filter)

# --- Collect expert trajectories ---
print("Sampling expert transitions.")
rollouts = rollout.rollout(
    expert,
    DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
    rollout.make_sample_until(min_timesteps=None, min_episodes=100),
    rng=rng,
    verbose=True,
    render=True,
)
# serialize.save("mpc_expert_rollouts.pkl", rollouts)
# -----------------------------------
