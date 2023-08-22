"""
This demo loads MPC rollouts as the expert and uses AIRL for imitation learning.
RUN COMMAND: python -m tests.test_mpc_airl
"""
import logging
import os

import numpy as np

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from imitation.data import serialize

from utils.utils import seed_everything
from utils.gym_utils import create_unified_flexiblearmenv_and_controller

logging.basicConfig(level=logging.INFO)
TRAIN_MODEL = True
SEED = 0
rng = np.random.default_rng(SEED)
seed_everything(SEED)

if TRAIN_MODEL:
    env, expert = create_unified_flexiblearmenv_and_controller(create_controller=True)
    venv = DummyVecEnv([lambda: env])

    # --- load expert rollouts ---
    rollouts = serialize.load("mpc_expert_rollouts.pkl")
    # ----------------------------

    learner = PPO(
        env=env,
        policy=MlpPolicy,
        batch_size=64,
        ent_coef=0.1,
        learning_rate=0.0001,
        n_epochs=5,
        seed=SEED,
    )
    reward_net = BasicShapedRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )
    airl_trainer = AIRL(
        demonstrations=rollouts,
        demo_batch_size=256,
        gen_replay_buffer_capacity=100000,
        n_disc_updates_per_round=2,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        debug_use_ground_truth=False,
    )

    # evaluate the learner before training
    env.reset(seed=SEED)
    learner_rewards_before_training, _ = evaluate_policy(
        model=learner,
        env=env,
        n_eval_episodes=3,
        return_episode_rewards=True,
        render=False,
    )
    print("mean reward before training:", np.mean(learner_rewards_before_training))

    # train the learner and evaluate again
    airl_trainer.train(10000)

    # save the trained model (create directory if needed)
    os.makedirs("trained_models", exist_ok=True)
    learner.save("trained_models/policy_mpc_airl.zip")
else:
    env, _ = create_unified_flexiblearmenv_and_controller(create_controller=False)
    learner = PPO.load("trained_models/policy_mpc_airl.zip")

# evaluate the learner after training
env.reset(seed=SEED)
learner_rewards_after_training, _ = evaluate_policy(
    model=learner,
    env=env,
    n_eval_episodes=3,
    return_episode_rewards=True,
    render=True,
)

print("mean reward after training:", np.mean(learner_rewards_after_training))
