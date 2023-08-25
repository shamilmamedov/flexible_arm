"""
This demo loads MPC rollouts as the expert and uses GAIL for imitation learning.
RUN COMMAND: python -m tests.test_mpc_gail
"""
import logging
import os

import numpy as np

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from imitation.data import serialize

from utils.utils import seed_everything
from utils.gym_utils import (
    create_unified_flexiblearmenv_and_controller_and_safety_filter,
)

logging.basicConfig(level=logging.INFO)
TRAIN_MODEL = True
SEED = 0
rng = np.random.default_rng(SEED)
seed_everything(SEED)

if TRAIN_MODEL:
    env, expert, _ = create_unified_flexiblearmenv_and_controller_and_safety_filter(
        create_controller=True, create_safety_filter=False
    )
    venv = DummyVecEnv([lambda: env])

    # --- load expert rollouts ---
    rollouts = serialize.load("mpc_expert_rollouts.pkl")
    # ----------------------------

    learner = PPO(
        env=env,
        policy=MlpPolicy,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.00001,
        n_epochs=1,
        seed=SEED,
        tensorboard_log="./logs/IRL/gail",
    )
    reward_net = BasicShapedRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )
    gail_trainer = GAIL(
        demonstrations=rollouts,
        demo_batch_size=1024,
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=4,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
    )

    # evaluate the learner before training
    env.reset(seed=SEED)
    learner_rewards_before_training, _ = evaluate_policy(
        model=learner,
        env=env,
        n_eval_episodes=30,
        return_episode_rewards=True,
        render=False,
    )
    print("mean reward before training:", np.mean(learner_rewards_before_training))

    # train the learner and evaluate again
    gail_trainer.train(100000)

    # save the trained model (create directory if needed)
    os.makedirs("trained_models", exist_ok=True)
    learner.save("trained_models/policy_mpc_gail.zip")
else:
    env, _, _ = create_unified_flexiblearmenv_and_controller_and_safety_filter(
        create_controller=False, create_safety_filter=False
    )
    learner = PPO.load("trained_models/policy_mpc_gail.zip")

# evaluate the learner after training
env.reset(seed=SEED)
learner_rewards_after_training, _ = evaluate_policy(
    model=learner,
    env=env,
    n_eval_episodes=30,
    return_episode_rewards=True,
    render=False,
)

print("mean reward after training:", np.mean(learner_rewards_after_training))


# plot histogram of rewards before and after training
if TRAIN_MODEL:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.hist(
        [learner_rewards_before_training, learner_rewards_after_training],
        label=["untrained", "trained"],
    )
    ax.legend()
    fig.savefig("trained_models/policy_mpc_gail_reward_histogram.png")
    plt.show()
