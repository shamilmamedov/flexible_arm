"""
This demo trains a PPO agent on the flexible arm environment.
"""

import logging
import os

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import ActorCriticPolicy
from stable_baselines3.common.evaluation import evaluate_policy

from utils.utils import seed_everything
from utils.gym_utils import create_unified_flexiblearmenv_and_controller

logging.basicConfig(level=logging.INFO)
TRAIN_MODEL = True
SEED = 0
seed_everything(SEED)

env, _ = create_unified_flexiblearmenv_and_controller(create_controller=False)
eval_env, _ = create_unified_flexiblearmenv_and_controller(create_controller=False)

if TRAIN_MODEL:
    logging.info("Training an PPO model")
    agent = PPO(
        policy=ActorCriticPolicy,
        env=env,
        verbose=1,
        seed=SEED,
        tensorboard_log="./logs/RL",
    )

    # evaluate the policy before training
    eval_env.reset(seed=SEED)
    reward_before, _ = evaluate_policy(
        model=agent, env=eval_env, n_eval_episodes=3, render=False
    )
    logging.info(f"Reward before training: {reward_before}")

    # train the agent
    agent.learn(total_timesteps=1000000)

    # save the trained policy
    os.makedirs("trained_models", exist_ok=True)
    agent.save("trained_models/policy_ppo.zip")
else:
    # load the trained policy
    agent = PPO.load("trained_models/policy_ppo.zip")

# evaluate the policy after training
eval_env.reset(seed=SEED)
reward_after, _ = evaluate_policy(
    model=agent, env=eval_env, n_eval_episodes=3, render=True
)
logging.info(f"Reward after training: {reward_after}")
