"""
This demo uses MPC as the expert collects data for the FlexibleArmEnv environment.
RUN COMMAND: python -m tests.test_mpc_data_collection
"""

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
import logging
logging.basicConfig(level=logging.INFO, filename="py_log.log",filemode="w")
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data import serialize
from utils.gym_utils import (
    create_unified_flexiblearmenv_and_controller_and_safety_filter,
)

SEED = 0
rng = np.random.default_rng(SEED)
USE_OBSTACLE = True
N_EPS = 1
SEG_LIST = [1,2,3,5]
logging.info(f"_______________________________________")
logging.info(f"Starting runs: with {N_EPS} episodes...")


n_seg_mpc_1_list = [3]
n_seg_mpc_2_list = [1]
n_seg_ratio_list = [.3]

for n_seg_mpc_1,n_seg_mpc_2,n_seg_ratio in zip(n_seg_mpc_1_list,n_seg_mpc_2_list,n_seg_ratio_list):
    np.random.seed(SEED)
    logging.info(f"Starting phases. n_seg_mpc_1: {n_seg_mpc_1}, n_seg_mpc_2: {n_seg_mpc_2}, n_seg_ratio: {n_seg_ratio}...")
    env, expert, _ = create_unified_flexiblearmenv_and_controller_and_safety_filter(
        controller_type="mpc_phases",
        create_controller=True,
        add_wall_obstacle=USE_OBSTACLE,
        create_safety_filter=False,
        n_seg_mpc=n_seg_mpc_1,
        n_seg_phases=(n_seg_mpc_1, n_seg_mpc_2),
        n_seg_ratio_p1=n_seg_ratio
        #   env_opts=env_options
    )

    # --- Collect expert trajectories ---
    print("Sampling phases transitions.")
    rollouts = rollout.rollout(
        expert,
        DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
        rollout.make_sample_until(min_timesteps=None, min_episodes=N_EPS),
        rng=rng,
        verbose=True,
        render=False,
    )
    serialize.save("mpc_expert_phases_rollouts_" + str(n_seg_mpc_1) +"_"+ str(n_seg_mpc_2) +".pkl", rollouts)
    print(
        f"Timings phases: mean={np.mean(expert.controller.debug_timings)}, std: {np.std(expert.controller.debug_timings)}")
    logging.info(f"Timings: mean={np.mean(expert.controller.debug_timings)}, std: {np.std(expert.controller.debug_timings)}")

for n_seg_mpc in SEG_LIST:
    # -----------------------------------
    print("Sampling MPC transitions with " + str(n_seg_mpc) + "-segments...")
    print("-----------------------------------")
    logging.info(f"Starting MPC. n_seg_mpc: {n_seg_mpc}...")

    np.random.seed(SEED)
    # env_options = {"flex_param_file_path": "/home/rudolf/PycharmProjects/flexible_arm/models/three_dof/one_segments/flexibility_params.yml"}
    env, expert, _ = create_unified_flexiblearmenv_and_controller_and_safety_filter(
        controller_type="mpc",
        create_controller=True,
        add_wall_obstacle=USE_OBSTACLE,
        create_safety_filter=False,
        n_seg_mpc=n_seg_mpc,
        #   env_opts=env_options
    )

    # --- Collect expert trajectories ---
    rollouts = rollout.rollout(
        expert,
        DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
        rollout.make_sample_until(min_timesteps=None, min_episodes=N_EPS),
        rng=rng,
        verbose=True,
        render=False,
    )
    serialize.save("mpc_expert_rollouts_" + str(n_seg_mpc) + ".pkl", rollouts)
    print(f"Timings: mean={np.mean(expert.controller.debug_timings)}, std: {np.std(expert.controller.debug_timings)}")
    logging.info(
        f"Timings: mean={np.mean(expert.controller.debug_timings)}, std: {np.std(expert.controller.debug_timings)}")
