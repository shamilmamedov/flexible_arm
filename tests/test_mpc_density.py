"""
This demo loads MPC rollouts as the expert and uses a Density based method for imitation learning.
RUN COMMAND: python -m tests.test_mpc_density
"""
import logging
import os

import numpy as np
import torch

from imitation.algorithms import density
from imitation.data import rollout
from imitation.data import serialize

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy

from utils.utils import seed_everything
from utils.gym_utils import create_unified_flexiblearmenv_and_controller

logging.basicConfig(level=logging.INFO)
TRAIN_MODEL = True
SEED = 0
rng = np.random.default_rng(SEED)
seed_everything(SEED)

env, _ = create_unified_flexiblearmenv_and_controller(create_controller=False)
venv = DummyVecEnv([lambda: env])

if TRAIN_MODEL:
    logging.info("Training a Density model")

    rollouts = serialize.load("mpc_expert_rollouts.pkl")

    transitions = rollout.flatten_trajectories(rollouts)

    imitation_trainer = SAC(
        SACPolicy, env, verbose=1, seed=SEED, tensorboard_log="./logs/IRL/density"
    )
    density_trainer = density.DensityAlgorithm(
        venv=venv,
        demonstrations=rollouts,
        rl_algo=imitation_trainer,
        rng=rng,
    )

    density_trainer.train()  # train the density model

    env.reset(seed=SEED)
    reward, _ = evaluate_policy(
        density_trainer.policy,  # type: ignore[arg-type]
        env,
        n_eval_episodes=3,
        render=False,
    )
    print(f"Reward before training: {reward}")

    print("Training the imitation learning policy")
    density_trainer.train_policy(n_timesteps=100000)

    # save the trained policy
    policy = density_trainer.policy
    os.makedirs("trained_models", exist_ok=True)
    torch.save(policy, "trained_models/policy_mpc_density.pt")
else:
    logging.info("Loading a trained densitybased RL model")
    policy = torch.load("trained_models/policy_mpc_density.pt")

env.reset(seed=SEED)
reward, _ = evaluate_policy(
    policy,  # type: ignore[arg-type]
    env,
    n_eval_episodes=3,
    render=True,
)
print(f"Reward after training: {reward}")
