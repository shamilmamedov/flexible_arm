"""
This demo trains an SAC agent on the flexible arm environment.
"""

import logging
import os

from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from utils.utils import seed_everything
from utils.gym_utils import (
    create_unified_flexiblearmenv_and_controller_and_safety_filter,
)

logging.basicConfig(level=logging.INFO)
TRAIN_MODEL = False
SEED = 0
seed_everything(SEED)

env, _, _ = create_unified_flexiblearmenv_and_controller_and_safety_filter(
    create_controller=False
)
eval_env, _, _ = create_unified_flexiblearmenv_and_controller_and_safety_filter(
    create_controller=False
)

if TRAIN_MODEL:
    logging.info("Training an SAC model")
    agent = SAC(
        policy=SACPolicy, env=env, verbose=1, seed=SEED, tensorboard_log="./logs/RL"
    )

    # evaluate the policy before training
    eval_env.reset(seed=SEED)
    reward_before, _ = evaluate_policy(
        model=agent, env=eval_env, n_eval_episodes=3, render=False
    )
    logging.info(f"Reward before training: {reward_before}")

    # train the agent
    agent.learn(total_timesteps=2000000)

    # save the trained policy
    os.makedirs("trained_models", exist_ok=True)
    agent.save("trained_models/policy_sac.zip")
else:
    # load the trained policy
    agent = SAC.load("trained_models/policy_sac.zip")

# evaluate the policy after training
eval_env.reset(seed=SEED)
reward_after, _ = evaluate_policy(
    model=agent, env=eval_env, n_eval_episodes=3, render=True
)
logging.info(f"Reward after training: {reward_after}")
