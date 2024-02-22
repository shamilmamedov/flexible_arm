"""
This demo trains an SAC agent on the flexible arm environment.
"""

import os
import sys
import logging
from datetime import datetime

from hydra import compose, initialize

from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

from utils.utils import seed_everything
from utils.gym_utils import (
    create_unified_flexiblearmenv_and_controller_and_safety_filter,
)

# Get hydra config
initialize(version_base=None, config_path="../conf", job_name="FlexibleArm")
cfg = compose(config_name="config", overrides=sys.argv[1:])

logging.basicConfig(level=logging.INFO)
TRAIN_MODEL = cfg.training.train
SEED = cfg.training.seed
DEVICE = cfg.training.device
TRAIN_STEPS = cfg.training.train_steps

now = datetime.now()
LOG_DIR = f"logs/RL/SAC/{now.strftime('%Y-%m-%d_%H-%M')}/SEED_{SEED}"
MODEL_DIR = f"trained_models/RL/SAC/{now.strftime('%Y-%m-%d_%H-%M')}/SEED_{SEED}"

seed_everything(SEED)

env, _, _ = create_unified_flexiblearmenv_and_controller_and_safety_filter(
    create_controller=False, add_wall_obstacle=True
)
eval_env, _, _ = create_unified_flexiblearmenv_and_controller_and_safety_filter(
    create_controller=False, add_wall_obstacle=True
)

if TRAIN_MODEL:
    logging.info("Training an SAC model from scratch")
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env=eval_env,
        n_eval_episodes=3,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=5000,
        deterministic=True,
        render=False,
    )
    agent = SAC(
        policy=SACPolicy,
        env=env,
        verbose=1,
        seed=SEED,
        tensorboard_log=LOG_DIR,
        learning_rate=0.0003,
        device=f"cuda:{DEVICE}",
    )

    # evaluate the policy before training
    eval_env.reset(seed=SEED)
    reward_before, _ = evaluate_policy(
        model=agent,
        env=eval_env,
        n_eval_episodes=3,
        render=False,
    )
    logging.info(f"Reward before training: {reward_before}")

    # train the agent
    agent.learn(total_timesteps=TRAIN_STEPS, callback=eval_callback)

    # save the trained policy
    agent.save(f"{MODEL_DIR}/policy_sac_last")
else:
    # load the trained policy
    agent = SAC.load(f"trained_models/RL/SAC/2023-08-30_14-14/best_model.zip")

# evaluate the policy after training
eval_env.reset(seed=SEED)
reward_after, _ = evaluate_policy(
    model=agent,
    env=eval_env,
    n_eval_episodes=3,
    render=False,
)
logging.info(f"Reward after training: {reward_after}")
