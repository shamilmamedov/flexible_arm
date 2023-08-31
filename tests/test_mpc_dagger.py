"""
This demo loads MPC rollouts as the expert and uses DAGGER for imitation learning.
RUN COMMAND: python -m tests.test_mpc_dagger
"""
import os
import logging
import tempfile
from datetime import datetime

import torch
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import configure_logger
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import bc
from imitation.data import serialize
from imitation.algorithms.dagger import SimpleDAggerTrainer

from utils.utils import seed_everything
from utils.gym_utils import (
    create_unified_flexiblearmenv_and_controller_and_safety_filter,
)

logging.basicConfig(level=logging.INFO)
TRAIN_MODEL = True
SEED = 0

now = datetime.now()
LOG_DIR = f"logs/IL/DAGGER/{now.strftime('%Y-%m-%d_%H-%M')}"
MODEL_DIR = f"trained_models/IL/DAGGER/{now.strftime('%Y-%m-%d_%H-%M')}"

rng = np.random.default_rng(SEED)
seed_everything(SEED)


env, expert, _ = create_unified_flexiblearmenv_and_controller_and_safety_filter(
    create_controller=True, create_safety_filter=False, add_wall_obstacle=True
)
eval_env, _, _ = create_unified_flexiblearmenv_and_controller_and_safety_filter(
    create_controller=False, create_safety_filter=False, add_wall_obstacle=True
)
if TRAIN_MODEL:
    logging.info("Training a DAGGER model from scratch")
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    venv = DummyVecEnv([lambda: env])
    custom_logger = configure_logger(
        verbose=1,
        tensorboard_log=LOG_DIR,
        reset_num_timesteps=True,
        tb_log_name="DAGGER",
    )
    policy = ActorCriticPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: torch.finfo(torch.float32).max,
        net_arch=[32, 32],
    )
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        policy=policy,
        rng=rng,
        custom_logger=custom_logger,
    )
    eval_callback = bc.BCEvalCallback(
        eval_env=eval_env,
        model=policy,
        eval_freq=1000,  # smaller than n_batches in bc.train()
        n_eval_episodes=3,
        logger=custom_logger,
        verbose=1,
        best_model_save_path=MODEL_DIR,
        log_dir=LOG_DIR,
        render=False,
    )

    rollouts = serialize.load("mpc_expert_rollouts.pkl")

    with tempfile.TemporaryDirectory(prefix="dagger_trained_") as tmpdir:
        dagger_trainer = SimpleDAggerTrainer(
            venv=venv,
            scratch_dir=tmpdir,
            expert_policy=expert,
            bc_trainer=bc_trainer,
            rng=rng,
            expert_trajs=rollouts,
        )

        eval_env.reset(seed=SEED)
        reward, _ = evaluate_policy(
            model=dagger_trainer.policy,  # type: ignore[arg-type]
            env=eval_env,
            n_eval_episodes=3,
            render=False,
        )
        print(f"Reward before training: {reward}")

        print("Training a policy using Dagger")
        dagger_trainer.train(
            1000000,
            bc_train_kwargs={
                "on_batch_end": eval_callback,
                "n_batches": 10000,
                "log_interval": 1000,
            },
        )

        policy = dagger_trainer.policy
        torch.save(policy, f"{MODEL_DIR}/policy_mpc_dagger_last.pt")

else:
    logging.info("Loading a trained DAGGER model")
    policy = torch.load("trained_models/IL/DAGGER/policy_mpc_dagger_last.pt")

eval_env.reset(seed=SEED)
reward, _ = evaluate_policy(
    model=policy,  # type: ignore[arg-type]
    env=eval_env,
    n_eval_episodes=3,
    render=False,
)
print(f"Reward after training: {reward}")
