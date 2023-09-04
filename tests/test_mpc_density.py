"""
This demo loads MPC rollouts as the expert and uses a Density based method for imitation learning.
RUN COMMAND: python -m tests.test_mpc_density
"""
import os
import logging
from datetime import datetime

import numpy as np

from imitation.data import serialize
from imitation.algorithms import density

from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import configure_logger
from stable_baselines3.common.evaluation import evaluate_policy

from utils.utils import seed_everything
from utils.gym_utils import (
    create_unified_flexiblearmenv_and_controller_and_safety_filter,
)

logging.basicConfig(level=logging.INFO)
TRAIN_MODEL = True
SEED = 0

now = datetime.now()
LOG_DIR = f"logs/IRL/DENSITY/{now.strftime('%Y-%m-%d_%H-%M')}"
MODEL_DIR = f"trained_models/IRL/DENSITY/{now.strftime('%Y-%m-%d_%H-%M')}/SEED_{SEED}"

rng = np.random.default_rng(SEED)
seed_everything(SEED)

env, expert, _ = create_unified_flexiblearmenv_and_controller_and_safety_filter(
    create_controller=True, create_safety_filter=False, add_wall_obstacle=True
)
eval_env, _, _ = create_unified_flexiblearmenv_and_controller_and_safety_filter(
    create_controller=False, create_safety_filter=False, add_wall_obstacle=True
)

if TRAIN_MODEL:
    logging.info("Training a Density model from scratch")
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    venv = DummyVecEnv([lambda: env])

    rollouts = serialize.load("mpc_expert_rollouts.pkl")

    eval_callback = EvalCallback(
        eval_env=eval_env,
        n_eval_episodes=1,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=10000,
        deterministic=True,
        render=False,
    )
    rl_algo = SAC(
        policy=SACPolicy,
        env=env,
        verbose=1,
        seed=SEED,
        tensorboard_log=LOG_DIR,
    )

    custom_logger = configure_logger(
        verbose=rl_algo.verbose,
        tensorboard_log=rl_algo.tensorboard_log,
        reset_num_timesteps=True,
        tb_log_name=rl_algo.__class__.__name__,
    )
    density_trainer = density.DensityAlgorithm(
        demonstrations=rollouts,
        venv=venv,
        rng=rng,
        rl_algo=rl_algo,
        rl_algo_callback=eval_callback,
        custom_logger=custom_logger,
    )

    density_trainer.train()  # train the density model

    eval_env.reset(seed=SEED)
    reward, _ = evaluate_policy(
        model=density_trainer.rl_algo,  # type: ignore[arg-type]
        env=eval_env,
        n_eval_episodes=3,
        render=False,
    )
    print(f"Reward before training: {reward}")

    print("Training the imitation learning policy")
    density_trainer.train_policy(n_timesteps=2000000)

    # save the trained policy
    policy = density_trainer.rl_algo
    policy.save(f"{MODEL_DIR}/policy_mpc_density_sac_last")
else:
    logging.info("Loading a trained densitybased RL model")
    policy = SAC.load(f"{MODEL_DIR}/policy_mpc_density_sac_last")

eval_env.reset(seed=SEED)
reward, _ = evaluate_policy(
    model=policy,  # type: ignore[arg-type]
    env=eval_env,
    n_eval_episodes=3,
    render=False,
)
print(f"Reward after training: {reward}")
