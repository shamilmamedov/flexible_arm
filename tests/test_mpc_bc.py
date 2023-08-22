"""
This demo loads MPC rollouts as the expert and uses BC for imitation learning.
RUN COMMAND: python -m tests.test_mpc_bc
"""
import logging
import os

import numpy as np
import torch
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data import serialize

from utils.utils import seed_everything
from utils.gym_utils import create_unified_flexiblearmenv_and_controller

logging.basicConfig(level=logging.INFO)
TRAIN_MODEL = True
SEED = 0
rng = np.random.default_rng(SEED)
seed_everything(SEED)

env, _ = create_unified_flexiblearmenv_and_controller(return_controller=False)

if TRAIN_MODEL:
    logging.info("Training a BC model")

    rollouts = serialize.load("mpc_expert_rollouts.pkl")

    transitions = rollout.flatten_trajectories(rollouts)

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=rng,
    )

    env.reset(seed=SEED)
    reward, _ = evaluate_policy(
        bc_trainer.policy,  # type: ignore[arg-type]
        env,
        n_eval_episodes=3,
        render=False,
    )
    print(f"Reward before training: {reward}")

    print("Training a policy using Behavior Cloning")
    bc_trainer.train(n_epochs=10)

    # save the trained policy
    policy = bc_trainer.policy
    os.makedirs("trained_models", exist_ok=True)
    torch.save(policy, "trained_models/policy_mpc_bc.pt")
else:
    logging.info("Loading a trained BC model")
    policy = torch.load("trained_models/policy_mpc_bc.pt")

env.reset(seed=SEED)
reward, _ = evaluate_policy(
    policy,  # type: ignore[arg-type]
    env,
    n_eval_episodes=3,
    render=True,
)
print(f"Reward after training: {reward}")
