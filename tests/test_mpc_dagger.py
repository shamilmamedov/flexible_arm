"""
This demo loads MPC rollouts as the expert and uses DAGGER for imitation learning.
RUN COMMAND: python -m tests.test_mpc_dagger
"""
import logging
import tempfile
import os

import torch
import numpy as np

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.data import serialize

from utils.utils import seed_everything
from utils.gym_utils import create_unified_flexiblearmenv_and_controller

logging.basicConfig(level=logging.INFO)
TRAIN_MODEL = True
SEED = 0
rng = np.random.default_rng(SEED)
seed_everything(SEED)

if TRAIN_MODEL:
    logging.info("Training a DAGGER model")
    env, expert = create_unified_flexiblearmenv_and_controller(create_controller=True)
    venv = DummyVecEnv([lambda: env])

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        rng=rng,
    )

    rollouts = serialize.load("mpc_expert_rollouts.pkl")

    with tempfile.TemporaryDirectory(prefix="dagger_trained_") as tmpdir:
        print(tmpdir)
        dagger_trainer = SimpleDAggerTrainer(
            venv=venv,
            scratch_dir=tmpdir,
            expert_policy=expert,
            bc_trainer=bc_trainer,
            rng=rng,
            expert_trajs=rollouts,
        )

        env.reset(seed=SEED)
        reward, _ = evaluate_policy(
            dagger_trainer.policy,  # type: ignore[arg-type]
            env,
            n_eval_episodes=3,
            render=False,
        )
        print(f"Reward before training: {reward}")

        print("Training a policy using Dagger")
        dagger_trainer.train(5000)

        policy = dagger_trainer.policy
        os.makedirs("trained_models", exist_ok=True)
        torch.save(policy, "trained_models/policy_mpc_dagger.pt")

else:
    env, _ = create_unified_flexiblearmenv_and_controller(return_controller=False)
    logging.info("Loading a trained DAGGER model")
    policy = torch.load("trained_models/policy_mpc_dagger.pt")

env.reset(seed=SEED)
reward, _ = evaluate_policy(
    policy,  # type: ignore[arg-type]
    env,
    n_eval_episodes=3,
    render=True,
)
print(f"Reward after training: {reward}")
