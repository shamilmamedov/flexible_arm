"""
This demo loads MPC rollouts as the expert and uses BC for imitation learning.
RUN COMMAND: python -m tests.test_mpc_bc
"""
import os
import sys
import logging
from datetime import datetime

import torch
import numpy as np
from hydra import compose, initialize

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import configure_logger
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data import serialize

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

now = datetime.now()
LOG_DIR = f"logs/IL/BC/{now.strftime('%Y-%m-%d_%H-%M')}/SEED_{SEED}"
MODEL_DIR = f"trained_models/IL/BC/{now.strftime('%Y-%m-%d_%H-%M')}/SEED_{SEED}"

rng = np.random.default_rng(SEED)
seed_everything(SEED)

env, expert, _ = create_unified_flexiblearmenv_and_controller_and_safety_filter(
    create_controller=True, create_safety_filter=False, add_wall_obstacle=True
)
eval_env, _, _ = create_unified_flexiblearmenv_and_controller_and_safety_filter(
    create_controller=False, create_safety_filter=False, add_wall_obstacle=True
)
if TRAIN_MODEL:
    logging.info("Training a BC model from scratch")
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    rollouts = serialize.load("demos/mpc_expert_rollouts.pkl")

    transitions = rollout.flatten_trajectories(rollouts)

    custom_logger = configure_logger(
        verbose=1,
        tensorboard_log=LOG_DIR,
        reset_num_timesteps=True,
        tb_log_name="BC",
    )

    policy = ActorCriticPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: torch.finfo(torch.float32).max,
        net_arch=[32, 32],
    )

    eval_callback = bc.BCEvalCallback(
        eval_env=eval_env,
        model=policy,
        eval_freq=5000,
        n_eval_episodes=3,
        logger=custom_logger,
        verbose=1,
        best_model_save_path=MODEL_DIR,
        log_dir=LOG_DIR,
        render=False,
    )
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        policy=policy,
        demonstrations=transitions,
        rng=rng,
        custom_logger=custom_logger,
        device=f"cuda:{DEVICE}",
    )
    eval_env.reset(seed=SEED)
    reward, _ = evaluate_policy(
        bc_trainer.policy,  # type: ignore[arg-type]
        eval_env,
        n_eval_episodes=3,
        render=False,
    )
    logging.info(f"Reward before training: {reward}")

    logging.info("Training a policy using Behavior Cloning")
    bc_trainer.train(
        n_batches=2000000,
        log_interval=100,
        on_batch_end=eval_callback,
    )

    # save the trained policy
    policy = bc_trainer.policy
    torch.save(policy, f"{MODEL_DIR}/policy_mpc_bc_last.pt")
else:
    logging.info("Loading a trained BC model")
    policy = torch.load("trained_models/IL/BC/2023-08-31_14-15/best_model.pt")

eval_env.reset(seed=SEED)
reward, _ = evaluate_policy(
    policy,  # type: ignore[arg-type]
    eval_env,
    n_eval_episodes=3,
    render=True,
)
logging.info(f"Reward after training: {reward}")
