"""
This demo trains a PPO agent on the flexible arm environment.
"""
import os
import logging
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import ActorCriticPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

from utils.utils import seed_everything
from utils.gym_utils import (
    create_unified_flexiblearmenv_and_controller_and_safety_filter,
)

logging.basicConfig(level=logging.INFO)
TRAIN_MODEL = True
SEED = 0

now = datetime.now()
LOG_DIR = f"logs/RL/PPO/{now.strftime('%Y-%m-%d_%H-%M')}"
MODEL_DIR = f"trained_models/RL/PPO/{now.strftime('%Y-%m-%d_%H-%M')}"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

seed_everything(SEED)

env, _, _ = create_unified_flexiblearmenv_and_controller_and_safety_filter(
    create_controller=False
)
eval_env, _, _ = create_unified_flexiblearmenv_and_controller_and_safety_filter(
    create_controller=False
)

if TRAIN_MODEL:
    logging.info("Training an PPO model")
    eval_callback = EvalCallback(
        eval_env=eval_env,
        n_eval_episodes=3,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=10000,
        deterministic=True,
        render=False,
    )
    agent = PPO(
        policy=ActorCriticPolicy,
        env=env,
        verbose=1,
        seed=SEED,
        tensorboard_log=LOG_DIR,
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
    agent.save(f"{MODEL_DIR}/policy_ppo_last")
else:
    # load the trained policy
    agent = PPO.load(f"{MODEL_DIR}/policy_ppo_last")

# evaluate the policy after training
eval_env.reset(seed=SEED)
reward_after, _ = evaluate_policy(
    model=agent, env=eval_env, n_eval_episodes=3, render=True
)
logging.info(f"Reward after training: {reward_after}")
