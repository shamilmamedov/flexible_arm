"""
Collects trajectories from each algorithm and saves them to a file. 
Then uses those trajectories to measure various KPIs.
RUN COMMAND: python -m tests.test_kpi
"""
import sys

import torch
import numpy as np
from hydra import compose, initialize

from imitation.data import rollout, serialize

from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from utils.utils import seed_everything
from utils.gym_utils import (
    create_unified_flexiblearmenv_and_controller_and_safety_filter,
)

# Get hydra config
initialize(version_base=None, config_path="../conf", job_name="FlexibleArm")
cfg = compose(config_name="config", overrides=sys.argv[1:])

DEMO_DIR = cfg.kpi.demo_dir
SEED = cfg.kpi.seed
rng = np.random.default_rng(SEED)
seed_everything(SEED)


if cfg.kpi.collect_demos:
    env, _, _ = create_unified_flexiblearmenv_and_controller_and_safety_filter(
        create_controller=False, add_wall_obstacle=True, create_safety_filter=False
    )
    venv = DummyVecEnv([lambda: env])

    if cfg.kpi.collect_bc:
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

else:
    bc_rollouts = serialize.load(f"{DEMO_DIR}/bc.pkl")
    dagger_rollouts = serialize.load(f"{DEMO_DIR}/dagger.pkl")
    gail_rollouts = serialize.load(f"{DEMO_DIR}/gail.pkl")
    airl_rollouts = serialize.load(f"{DEMO_DIR}/airl.pkl")
    density_rollouts = serialize.load(f"{DEMO_DIR}/density.pkl")
    sac_rollouts = serialize.load(f"{DEMO_DIR}/sac.pkl")
    ppo_rollouts = serialize.load(f"{DEMO_DIR}/ppo.pkl")


# Measure KPIs
# TODO: MEASURE KPIs FOR EACH ALGORITHM
# TODO: PLOT NECESSARY GRAPHS
