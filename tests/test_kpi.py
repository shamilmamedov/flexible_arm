"""
Collects trajectories from each algorithm and saves them to a file. 
Then uses those trajectories to measure various KPIs.
RUN COMMAND: python -m tests.test_kpi
"""
import os
import sys
import logging

import torch
import numpy as np
from hydra import compose, initialize
from omegaconf import OmegaConf

import matplotlib.pyplot as plt

from imitation.data import rollout, serialize

from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt

from utils.utils import seed_everything
from utils.gym_utils import (
    create_unified_flexiblearmenv_and_controller_and_safety_filter,
    SafetyWrapper,
)
from kpi import steps2reach_goal, path_length, constraint_violation, trajectory_reward

# Get hydra config
initialize(version_base=None, config_path="../conf", job_name="FlexibleArm")
cfg = compose(config_name="config", overrides=sys.argv[1:])
print(OmegaConf.to_yaml(cfg))

logging.basicConfig(level=logging.INFO)
DEMO_DIR = cfg.kpi.demo_dir
PLOT_DIR = cfg.kpi.plot_dir
SEED = cfg.kpi.seed
rng = np.random.default_rng(SEED)
seed_everything(SEED)

if cfg.kpi.random_goal:
    DEMO_DIR = f"{DEMO_DIR}/random_goal"
    env_options_override = None
else:
    DEMO_DIR = f"{DEMO_DIR}/near_wall_goal"
    env_options_override = {
        "qa_goal_start": np.array([-np.pi / 12, 0.0, -np.pi + 0.2]),
        "qa_goal_end": np.array([-np.pi / 12, np.pi / 2, 0.0]),
    }
os.makedirs(DEMO_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

if cfg.kpi.collect_demos:
    (
        env,
        expert,
        safety_filter,
    ) = create_unified_flexiblearmenv_and_controller_and_safety_filter(
        create_controller=True,
        add_wall_obstacle=True,
        create_safety_filter=True,
        env_opts=env_options_override,
    )
    venv = DummyVecEnv([lambda: env])

    if cfg.kpi.collect_bc:
        logging.info("Collecting BC rollouts")
        bc_model = torch.load(cfg.kpi.bc_model_path)
        bc_rollouts = rollout.rollout(
            policy=bc_model,
            venv=venv,
            sample_until=rollout.make_sample_until(min_episodes=cfg.kpi.n_demos),
            rng=rng,
            unwrap=False,
            exclude_infos=True,
            verbose=True,
            render=cfg.kpi.render,
        )
        serialize.save(f"{DEMO_DIR}/bc.pkl", bc_rollouts)

    if cfg.kpi.collect_dagger:
        logging.info("Collecting DAGGER rollouts")
        dagger_model = torch.load(cfg.kpi.dagger_model_path)
        dagger_rollouts = rollout.rollout(
            policy=dagger_model,
            venv=venv,
            sample_until=rollout.make_sample_until(min_episodes=cfg.kpi.n_demos),
            rng=rng,
            unwrap=False,
            exclude_infos=True,
            verbose=True,
            render=cfg.kpi.render,
        )
        serialize.save(f"{DEMO_DIR}/dagger.pkl", dagger_rollouts)

    if cfg.kpi.collect_gail:
        logging.info("Collecting GAIL rollouts")
        gail_model = SAC.load(cfg.kpi.gail_model_path)
        gail_rollouts = rollout.rollout(
            policy=gail_model,
            venv=venv,
            sample_until=rollout.make_sample_until(min_episodes=cfg.kpi.n_demos),
            rng=rng,
            unwrap=False,
            exclude_infos=True,
            verbose=True,
            render=cfg.kpi.render,
        )
        serialize.save(f"{DEMO_DIR}/gail.pkl", gail_rollouts)

    if cfg.kpi.collect_airl:
        logging.info("Collecting AIRL rollouts")
        airl_model = SAC.load(cfg.kpi.airl_model_path)
        airl_rollouts = rollout.rollout(
            policy=airl_model,
            venv=venv,
            sample_until=rollout.make_sample_until(min_episodes=cfg.kpi.n_demos),
            rng=rng,
            unwrap=False,
            exclude_infos=True,
            verbose=True,
            render=cfg.kpi.render,
        )
        serialize.save(f"{DEMO_DIR}/airl.pkl", airl_rollouts)

    if cfg.kpi.collect_density:
        logging.info("Collecting Density rollouts")
        density_model = SAC.load(cfg.kpi.density_model_path)
        density_rollouts = rollout.rollout(
            policy=density_model,
            venv=venv,
            sample_until=rollout.make_sample_until(min_episodes=cfg.kpi.n_demos),
            rng=rng,
            unwrap=False,
            exclude_infos=True,
            verbose=True,
            render=cfg.kpi.render,
        )
        serialize.save(f"{DEMO_DIR}/density.pkl", density_rollouts)

    if cfg.kpi.collect_sac:
        logging.info("Collecting SAC rollouts")
        sac_model = SAC.load(cfg.kpi.sac_model_path)
        sac_rollouts = rollout.rollout(
            policy=sac_model,
            venv=venv,
            sample_until=rollout.make_sample_until(min_episodes=cfg.kpi.n_demos),
            rng=rng,
            unwrap=False,
            exclude_infos=True,
            verbose=True,
            render=cfg.kpi.render,
        )
        serialize.save(f"{DEMO_DIR}/sac.pkl", sac_rollouts)

    if cfg.kpi.collect_ppo:
        logging.info("Collecting PPO rollouts")
        ppo_model = PPO.load(cfg.kpi.ppo_model_path)
        ppo_rollouts = rollout.rollout(
            policy=ppo_model,
            venv=venv,
            sample_until=rollout.make_sample_until(min_episodes=cfg.kpi.n_demos),
            rng=rng,
            unwrap=False,
            exclude_infos=True,
            verbose=True,
            render=cfg.kpi.render,
        )
        serialize.save(f"{DEMO_DIR}/ppo.pkl", ppo_rollouts)

    if cfg.kpi.collect_sac_sf:
        logging.info("Collecting SAC Safety Filter rollouts")
        sac_model = SAC.load(cfg.kpi.sac_model_path)
        sac_sf_model = SafetyWrapper(
            policy=sac_model.policy, safety_filter=safety_filter
        )
        sac_sf_rollouts = rollout.rollout(
            policy=sac_sf_model,
            venv=venv,
            sample_until=rollout.make_sample_until(min_episodes=cfg.kpi.n_demos),
            rng=rng,
            unwrap=False,
            exclude_infos=True,
            verbose=True,
            render=cfg.kpi.render,
        )
        serialize.save(f"{DEMO_DIR}/sac_sf.pkl", sac_sf_rollouts)

    if cfg.kpi.collect_dagger_sf:
        logging.info("Collecting DAGGER Safety Filter rollouts")
        dagger_model = torch.load(cfg.kpi.dagger_model_path)
        dagger_sf_model = SafetyWrapper(
            policy=dagger_model, safety_filter=safety_filter
        )
        dagger_sf_rollouts = rollout.rollout(
            policy=dagger_sf_model,
            venv=venv,
            sample_until=rollout.make_sample_until(min_episodes=cfg.kpi.n_demos),
            rng=rng,
            unwrap=False,
            exclude_infos=True,
            verbose=True,
            render=cfg.kpi.render,
        )
        serialize.save(f"{DEMO_DIR}/dagger_sf.pkl", dagger_sf_rollouts)

    if cfg.kpi.collect_mpc:
        logging.info("Collecting MPC rollouts")
        mpc_rollouts = rollout.rollout(
            policy=expert,
            venv=venv,
            sample_until=rollout.make_sample_until(min_episodes=cfg.kpi.n_demos),
            rng=rng,
            unwrap=False,
            exclude_infos=True,
            verbose=True,
            render=cfg.kpi.render,
        )
        serialize.save(f"{DEMO_DIR}/mpc.pkl", mpc_rollouts)


def mean_kpis(rollouts):
    kpis = dict()
    # kpis["steps2reach_goal"] = np.mean(steps2reach_goal(rollouts, 0.1)) # TODO: hwo to deal with None values?
    kpis["path_length"] = np.mean(path_length(rollouts))
    kpis["constraint_violation"] = np.mean(constraint_violation(rollouts))
    kpis["trajectory_reward"] = np.mean(trajectory_reward(rollouts))
    return kpis


if cfg.kpi.reward_box_plot:
    # uses random goals
    bc_rollouts = serialize.load(f"{DEMO_DIR}/bc.pkl")
    dagger_rollouts = serialize.load(f"{DEMO_DIR}/dagger.pkl")
    gail_rollouts = serialize.load(f"{DEMO_DIR}/gail.pkl")
    airl_rollouts = serialize.load(f"{DEMO_DIR}/airl.pkl")
    density_rollouts = serialize.load(f"{DEMO_DIR}/density.pkl")
    sac_rollouts = serialize.load(f"{DEMO_DIR}/sac.pkl")
    ppo_rollouts = serialize.load(f"{DEMO_DIR}/ppo.pkl")

    # --- box plot of rewards ---
    sac_rewards = trajectory_reward(sac_rollouts)
    ppo_rewards = trajectory_reward(ppo_rollouts)
    bc_rewards = trajectory_reward(bc_rollouts)
    dagger_rewards = trajectory_reward(dagger_rollouts)
    gail_rewards = trajectory_reward(gail_rollouts)
    airl_rewards = trajectory_reward(airl_rollouts)
    density_rewards = trajectory_reward(density_rollouts)

    fig, ax = plt.subplots()
    ax.boxplot(
        [
            dagger_rewards,
            bc_rewards,
            sac_rewards,
            ppo_rewards,
            airl_rewards,
            gail_rewards,
            density_rewards,
        ],
        labels=["DAGGER", "BC", "SAC", "PPO", "AIRL", "GAIL", "DENSITY"],
        zorder=3,
        notch=True,
        patch_artist=True,
        boxprops=dict(facecolor="blueviolet"),
    )
    ax.set_facecolor("ghostwhite")
    ax.grid(color="white", linestyle="-", linewidth=1, zorder=0)
    ax.set_title("Trajectory Rewards", fontdict={"fontsize": 16})
    ax.set_ylabel("Reward", fontdict={"fontsize": 14})
    ax.xaxis.set_tick_params(labelsize=12)
    fig.savefig(f"{PLOT_DIR}/kpi_rewards.png")
    fig.savefig(f"{PLOT_DIR}/kpi_rewards.pdf")
    plt.show()

if cfg.kpi.reward_constraint_scatter_plot:
    # uses the goals near the wall
    sac_rollouts = serialize.load(f"{DEMO_DIR}/sac.pkl")
    dagger_rollouts = serialize.load(f"{DEMO_DIR}/dagger.pkl")
    sac_sf_rollouts = serialize.load(f"{DEMO_DIR}/sac_sf.pkl")
    dagger_sf_rollouts = serialize.load(f"{DEMO_DIR}/dagger_sf.pkl")
    mpc_rollouts = serialize.load(f"{DEMO_DIR}/mpc.pkl")

    sac_kpis = mean_kpis(sac_rollouts)
    sac_sf_kpis = mean_kpis(sac_sf_rollouts)
    dagger_kpis = mean_kpis(dagger_rollouts)
    dagger_sf_kpis = mean_kpis(dagger_sf_rollouts)
    mpc_kpis = mean_kpis(mpc_rollouts)

    fig, ax = plt.subplots()
    rewards = [
        sac_kpis["trajectory_reward"],
        sac_sf_kpis["trajectory_reward"],
        dagger_kpis["trajectory_reward"],
        dagger_sf_kpis["trajectory_reward"],
        mpc_kpis["trajectory_reward"],
    ]
    violations = [
        sac_kpis["constraint_violation"],
        sac_sf_kpis["constraint_violation"],
        dagger_kpis["constraint_violation"],
        dagger_sf_kpis["constraint_violation"],
        mpc_kpis["constraint_violation"],
    ]
    ax.scatter(violations, rewards, zorder=3, c="blueviolet", s=100)
    ax.set_facecolor("ghostwhite")
    ax.grid(color="white", linestyle="-", linewidth=1, zorder=0)

    # Annotate points
    for i, (txt, offset) in enumerate(
        zip(
            ["SAC", "SAC+SF", "DAGGER", "DAGGER+SF", "MPC"],
            [(-15, -30), (-25, 15), (-30, -30), (-20, -30), (-15, -30)],
        )
    ):
        ax.annotate(
            txt, (violations[i], rewards[i]), textcoords="offset pixels", xytext=offset
        )

    ax.set_title("Reward vs Constraint Violation", fontdict={"fontsize": 16})
    ax.set_xlabel("Constraint Violation", fontdict={"fontsize": 14})
    ax.set_ylabel("Reward", fontdict={"fontsize": 14})
    fig.savefig(f"{PLOT_DIR}/kpi_reward_constraint_scatter.png")
    fig.savefig(f"{PLOT_DIR}/kpi_reward_constraint_scatter.pdf")
    plt.show()

if cfg.kpi.time_constraint_scatter_plot:
    sac_rollouts = serialize.load(f"{DEMO_DIR}/sac.pkl")
    dagger_rollouts = serialize.load(f"{DEMO_DIR}/dagger.pkl")
    sac_sf_rollouts = serialize.load(f"{DEMO_DIR}/sac_sf.pkl")
    dagger_sf_rollouts = serialize.load(f"{DEMO_DIR}/dagger_sf.pkl")
    mpc_rollouts = serialize.load(f"{DEMO_DIR}/mpc.pkl")

    sac_kpis = mean_kpis(sac_rollouts)
    sac_sf_kpis = mean_kpis(sac_sf_rollouts)
    dagger_kpis = mean_kpis(dagger_rollouts)
    dagger_sf_kpis = mean_kpis(dagger_sf_rollouts)
    mpc_kpis = mean_kpis(mpc_rollouts)

    violations = [
        sac_kpis["constraint_violation"],
        sac_sf_kpis["constraint_violation"],
        dagger_kpis["constraint_violation"],
        dagger_sf_kpis["constraint_violation"],
        mpc_kpis["constraint_violation"],
    ]

    # from test_timing.py
    timings = {
        "SAC": 0.25,
        "SAC+SF": 22.1,
        "DAGGER": 0.21,
        "DAGGER+SF": 21.9,
        "MPC": 44.7,
    }
    timings = [*timings.values()]

    fig, ax = plt.subplots()
    ax.scatter(violations, timings, zorder=3, c="blueviolet", s=100)
    ax.set_facecolor("ghostwhite")
    ax.grid(color="white", linestyle="-", linewidth=1, zorder=0)

    # Annotate points
    for i, (txt, offset) in enumerate(
        zip(
            ["SAC", "SAC+SF", "DAGGER", "DAGGER+SF", "MPC"],
            [(-15, 10), (-25, 10), (-25, 10), (-20, -25), (-15, -20)],
        )
    ):
        ax.annotate(
            txt, (violations[i], timings[i]), textcoords="offset pixels", xytext=offset
        )

    ax.set_title("Inference Time vs Constraint Violation", fontdict={"fontsize": 16})
    ax.set_xlabel("Constraint Violation", fontdict={"fontsize": 14})
    ax.set_ylabel("Inference Time (ms)", fontdict={"fontsize": 14})
    fig.savefig(f"{PLOT_DIR}/kpi_time_constraint_scatter.png")
    fig.savefig(f"{PLOT_DIR}/kpi_time_constraint_scatter.pdf")
    plt.show()
