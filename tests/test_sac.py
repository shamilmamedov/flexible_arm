"""
This demo trains an SAC agent on the flexible arm environment.
"""
import os
import logging
from datetime import datetime

from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy
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
LOG_DIR = f"logs/RL/SAC/{now.strftime('%Y-%m-%d_%H-%M')}"
MODEL_DIR = f"trained_models/RL/SAC/{now.strftime('%Y-%m-%d_%H-%M')}"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

seed_everything(SEED)

env, _, _ = create_unified_flexiblearmenv_and_controller_and_safety_filter(
    create_controller=False, add_wall_obstacle=True
)
eval_env, _, _ = create_unified_flexiblearmenv_and_controller_and_safety_filter(
    create_controller=False, add_wall_obstacle=True
)

if TRAIN_MODEL:
    logging.info("Training an SAC model")

    eval_callback = EvalCallback(
        eval_env=eval_env,
        n_eval_episodes=3,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=10000,
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
    agent.learn(total_timesteps=2000000, callback=eval_callback)

    # save the trained policy
    agent.save(f"{MODEL_DIR}/policy_sac_last")
else:
    # load the trained policy
    agent = SAC.load(f"{MODEL_DIR}/policy_sac_last")

# evaluate the policy after training
eval_env.reset(seed=SEED)
reward_after, _ = evaluate_policy(
    model=agent,
    env=eval_env,
    n_eval_episodes=3,
    render=False,
)
logging.info(f"Reward after training: {reward_after}")
