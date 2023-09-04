"""
This demo loads MPC rollouts as the expert and uses GAIL for imitation learning.
RUN COMMAND: python -m tests.test_mpc_gail
"""
import os
import sys
import logging
from datetime import datetime

import numpy as np
from hydra import compose, initialize

from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import configure_logger
from stable_baselines3.common.evaluation import evaluate_policy


from imitation.data import serialize
from imitation.util.networks import RunningNorm
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicShapedRewardNet

from utils.utils import seed_everything
from utils.gym_utils import (
    create_unified_flexiblearmenv_and_controller_and_safety_filter,
)

# Get hydra config
initialize(version_base=None, config_path="../conf", job_name="FlexibleArm")
cfg = compose(config_name="config", overrides=sys.argv[1:])

logging.basicConfig(level=logging.INFO)
TRAIN_MODEL = cfg.train
SEED = cfg.seed
DEVICE = cfg.device

now = datetime.now()
LOG_DIR = f"logs/IRL/GAIL/{now.strftime('%Y-%m-%d_%H-%M')}/SEEED_{SEED}"
MODEL_DIR = f"trained_models/IRL/GAIL/{now.strftime('%Y-%m-%d_%H-%M')}/SEED_{SEED}"

rng = np.random.default_rng(SEED)
seed_everything(SEED)

env, expert, _ = create_unified_flexiblearmenv_and_controller_and_safety_filter(
    create_controller=True, create_safety_filter=False, add_wall_obstacle=True
)
eval_env, _, _ = create_unified_flexiblearmenv_and_controller_and_safety_filter(
    create_controller=False, create_safety_filter=False, add_wall_obstacle=True
)
if TRAIN_MODEL:
    logging.info("Training GAIL model from scratch")
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    venv = DummyVecEnv([lambda: env])

    # --- load expert rollouts ---
    rollouts = serialize.load("demos/mpc_expert_rollouts.pkl")
    # ----------------------------
    eval_callback = EvalCallback(
        eval_env=eval_env,
        n_eval_episodes=3,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=5000,
        deterministic=True,
        render=False,
    )
    learner = SAC(
        policy=SACPolicy,
        env=env,
        verbose=1,
        seed=SEED,
        tensorboard_log=LOG_DIR,
        device=f"cuda:{DEVICE}",
    )

    custom_logger = configure_logger(
        verbose=learner.verbose,
        tensorboard_log=learner.tensorboard_log,
        reset_num_timesteps=True,
        tb_log_name=learner.__class__.__name__,
    )

    reward_net = BasicShapedRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )
    gail_trainer = GAIL(
        demonstrations=rollouts,
        demo_batch_size=1024,
        gen_replay_buffer_capacity=1000000,
        n_disc_updates_per_round=2,
        venv=venv,
        gen_algo=learner,
        gen_train_timesteps=1,
        gen_algo_callback=eval_callback,
        custom_logger=custom_logger,
        reward_net=reward_net,
        init_tensorboard=True,
        log_dir=LOG_DIR,
        debug_use_ground_truth=False,
        allow_variable_horizon=True,
    )

    # evaluate the learner before training
    eval_env.reset(seed=SEED)
    learner_rewards_before_training, _ = evaluate_policy(
        model=learner,
        env=eval_env,
        n_eval_episodes=3,
        return_episode_rewards=True,
        render=False,
    )
    logging.info(
        "mean reward before training:", np.mean(learner_rewards_before_training)
    )

    # train the learner and evaluate again
    gail_trainer.train(2000000)

    # save the trained model (create directory if needed)
    learner.save(f"{MODEL_DIR}/policy_mpc_gail_last")
else:
    learner = SAC.load(f"{MODEL_DIR}/policy_mpc_gail_last")

# evaluate the learner after training
eval_env.reset(seed=SEED)
learner_rewards_after_training, _ = evaluate_policy(
    model=learner,
    env=eval_env,
    n_eval_episodes=3,
    return_episode_rewards=True,
    render=False,
)

logging.info("mean reward after training:", np.mean(learner_rewards_after_training))


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
