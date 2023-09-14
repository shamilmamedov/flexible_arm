"""
Collects trajectories from each algorithm and saves them to a file. 
Then uses those trajectories to measure various KPIs.
RUN COMMAND: python -m tests.test_kpi
"""
import os
import sys
import logging
from pathlib import Path

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
import matplotlib

from utils.utils import seed_everything
from utils.gym_utils import (
    create_unified_flexiblearmenv_and_controller_and_safety_filter,
    SafetyWrapper,
)
from kpi import (
    steps2reach_goal,
    path_length,
    constraint_violation,
    trajectory_final_distance,
)

# Get hydra config
initialize(version_base=None, config_path="../conf", job_name="FlexibleArm")
cfg = compose(config_name="config", overrides=sys.argv[1:])
print(OmegaConf.to_yaml(cfg))

logging.basicConfig(level=logging.INFO)
DEMO_DIR = Path(cfg.kpi.demo_dir)
PLOT_DIR = Path(cfg.kpi.plot_dir)
SEED = cfg.kpi.seed
rng = np.random.default_rng(SEED)
seed_everything(SEED)

env_options_override = dict()
if cfg.kpi.random_goal:
    DEMO_DIR = DEMO_DIR / "random_goal"
    PLOT_DIR = PLOT_DIR / "random_goal"
else:
    DEMO_DIR = DEMO_DIR / "near_wall_goal"
    PLOT_DIR = PLOT_DIR / "near_wall_goal"
    env_options_override = {
        "qa_goal_start": np.array([-np.pi / 12, 0.0, -np.pi + 0.2]),
        "qa_goal_end": np.array([-np.pi / 12, np.pi / 2, 0.0]),
    }

if cfg.kpi.more_flexible:
    DEMO_DIR = DEMO_DIR / "more_flexible"
    PLOT_DIR = PLOT_DIR / "more_flexible"
    env_options_override["flex_param_file_path"] = cfg.kpi.flex_param_file_path


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
    kpis["trajectory_final_reward"] = np.mean(trajectory_final_distance(rollouts))
    return kpis


# --- plot settings ---
params = {  #'backend': 'ps',
    "text.latex.preamble": r"\usepackage{gensymb} \usepackage{amsmath}",
    "axes.labelsize": 8,  # fontsize for x and y labels (was 10)
    "axes.titlesize": 8,
    "legend.fontsize": 6,
    "xtick.labelsize": 10,
    "ytick.labelsize": 12,
    "text.usetex": True,
    "font.family": "serif",
}
matplotlib.rcParams.update(params)


if cfg.kpi.distance_box_plot:
    # uses random goals
    bc_rollouts = serialize.load(f"{DEMO_DIR}/bc.pkl")
    dagger_rollouts = serialize.load(f"{DEMO_DIR}/dagger.pkl")
    gail_rollouts = serialize.load(f"{DEMO_DIR}/gail.pkl")
    airl_rollouts = serialize.load(f"{DEMO_DIR}/airl.pkl")
    density_rollouts = serialize.load(f"{DEMO_DIR}/density.pkl")
    sac_rollouts = serialize.load(f"{DEMO_DIR}/sac.pkl")
    ppo_rollouts = serialize.load(f"{DEMO_DIR}/ppo.pkl")
    mpc_rollouts = serialize.load(f"{DEMO_DIR}/mpc.pkl")

    # --- box plot of rewards ---
    sac_rewards = trajectory_final_distance(sac_rollouts)
    ppo_rewards = trajectory_final_distance(ppo_rollouts)
    bc_rewards = trajectory_final_distance(bc_rollouts)
    dagger_rewards = trajectory_final_distance(dagger_rollouts)
    gail_rewards = trajectory_final_distance(gail_rollouts)
    airl_rewards = trajectory_final_distance(airl_rollouts)
    density_rewards = trajectory_final_distance(density_rollouts)
    mpc_rewards = trajectory_final_distance(mpc_rollouts)

    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.boxplot(
        [
            mpc_rewards,
            dagger_rewards,
            bc_rewards,
            sac_rewards,
            ppo_rewards,
            airl_rewards,
            gail_rewards,
            density_rewards,
        ],
        labels=["MPC", "DAGGER", "BC", "SAC", "PPO", "AIRL", "GAIL", "DENSITY"],
        zorder=3,
        notch=False,
        patch_artist=True,
        boxprops=dict(facecolor="blueviolet"),
    )
    ax.set_facecolor("lavender")
    ax.grid(color="white", linestyle="-", linewidth=1, zorder=0)
    ax.set_ylabel("Final Distance to Goal (cm)", fontdict={"fontsize": 12})
    fig.savefig(f"{PLOT_DIR}/kpi_distance_box.png")
    fig.savefig(
        f"{PLOT_DIR}/kpi_distance_box.pdf", format="pdf", dpi=600, bbox_inches="tight"
    )
    plt.show()

if cfg.kpi.distance_constraint_time_scatter_plot:
    # uses the goals near the wall (both normal and flexible)
    if DEMO_DIR.name == "more_flexible":
        flex_dir = DEMO_DIR
        normal_dir = DEMO_DIR.parent
    else:
        flex_dir = DEMO_DIR / "more_flexible"
        normal_dir = DEMO_DIR
    # --- normal ---
    sac_rollouts = serialize.load(normal_dir / "sac.pkl")
    dagger_rollouts = serialize.load(normal_dir / "dagger.pkl")
    sac_sf_rollouts = serialize.load(normal_dir / "sac_sf.pkl")
    dagger_sf_rollouts = serialize.load(normal_dir / "dagger_sf.pkl")
    mpc_rollouts = serialize.load(normal_dir / "mpc.pkl")
    # --- flexible ---
    flex_sac_rollouts = serialize.load(flex_dir / "sac.pkl")
    flex_dagger_rollouts = serialize.load(flex_dir / "dagger.pkl")
    flex_sac_sf_rollouts = serialize.load(flex_dir / "sac_sf.pkl")
    flex_dagger_sf_rollouts = serialize.load(flex_dir / "dagger_sf.pkl")
    flex_mpc_rollouts = serialize.load(flex_dir / "mpc.pkl")

    # --- normal kpi ---
    sac_kpis = mean_kpis(sac_rollouts)
    sac_sf_kpis = mean_kpis(sac_sf_rollouts)
    dagger_kpis = mean_kpis(dagger_rollouts)
    dagger_sf_kpis = mean_kpis(dagger_sf_rollouts)
    mpc_kpis = mean_kpis(mpc_rollouts)
    # --- flexible kpi ---
    flex_sac_kpis = mean_kpis(flex_sac_rollouts)
    flex_sac_sf_kpis = mean_kpis(flex_sac_sf_rollouts)
    flex_dagger_kpis = mean_kpis(flex_dagger_rollouts)
    flex_dagger_sf_kpis = mean_kpis(flex_dagger_sf_rollouts)
    flex_mpc_kpis = mean_kpis(flex_mpc_rollouts)

    # from test_timing.py in (ms)
    timings = {
        "SAC": 0.25,
        "SAC+SF": 22.1,
        "DAGGER": 0.21,
        "DAGGER+SF": 21.9,
        "MPC": 44.7,
    }
    timings = [*timings.values()]

    fig, ax = plt.subplots(figsize=(6, 3))
    # --- normal ---
    rewards = [
        sac_kpis["trajectory_final_reward"],
        sac_sf_kpis["trajectory_final_reward"],
        dagger_kpis["trajectory_final_reward"],
        dagger_sf_kpis["trajectory_final_reward"],
        mpc_kpis["trajectory_final_reward"],
    ]
    violations = [
        sac_kpis["constraint_violation"],
        sac_sf_kpis["constraint_violation"],
        dagger_kpis["constraint_violation"],
        dagger_sf_kpis["constraint_violation"],
        mpc_kpis["constraint_violation"],
    ]
    graph = ax.scatter(
        violations,
        rewards,
        zorder=3,
        s=200,
        c=timings,
        cmap="cool",
        alpha=0.9,
    )
    # --- flexible ---
    flex_rewards = [
        flex_sac_kpis["trajectory_final_reward"],
        flex_sac_sf_kpis["trajectory_final_reward"],
        flex_dagger_kpis["trajectory_final_reward"],
        flex_dagger_sf_kpis["trajectory_final_reward"],
        flex_mpc_kpis["trajectory_final_reward"],
    ]
    flex_violations = [
        flex_sac_kpis["constraint_violation"],
        flex_sac_sf_kpis["constraint_violation"],
        flex_dagger_kpis["constraint_violation"],
        flex_dagger_sf_kpis["constraint_violation"],
        flex_mpc_kpis["constraint_violation"],
    ]
    graph = ax.scatter(
        flex_violations,
        flex_rewards,
        zorder=3,
        s=200,
        c=timings,
        cmap="cool",
        edgecolors="darkolivegreen",
        linewidths=3,
        alpha=0.9,
    )

    graph_cbar = fig.colorbar(mappable=graph, ax=ax)
    graph_cbar.set_label("Inference Time (ms)", fontdict={"fontsize": 12})
    ax.set_facecolor("lavender")
    ax.grid(color="white", linestyle="-", linewidth=1, zorder=0)
    ax.set_xlim(0.0, 40)
    ax.set_ylim(0.0, 100)
    ax.legend(
        ["Flexible", "More Flexible"],
        loc="upper right",
        fontsize=12,
        facecolor="lemonchiffon",
    )

    # Annotate points
    for i, (txt, offset) in enumerate(
        zip(
            ["SAC", "SAC+SF", "DAGGER", "DAGGER+SF", "MPC"],
            [(5, -25), (0, -25), (15, 15), (-5, 15), (0, 15)],
        )
    ):
        ax.annotate(
            txt, (violations[i], rewards[i]), textcoords="offset pixels", xytext=offset
        )

    # ax.set_title("Reward vs Constraint Violation", fontdict={"fontsize": 16})
    ax.set_xlabel("Constraint Violations", fontdict={"fontsize": 12})
    ax.set_ylabel("Final Distance to Goal (cm)", fontdict={"fontsize": 12})
    fig.savefig(f"{PLOT_DIR}/kpi_distance_constraint_time_scatter_with_flex.png")
    fig.savefig(f"{PLOT_DIR}/kpi_distance_constraint_time_scatter_with_flex.pdf")
    plt.show()
