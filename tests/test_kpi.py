"""
Collects trajectories from each algorithm and saves them to a file. 
Then uses those trajectories to measure various KPIs.
RUN COMMAND: python -m tests.test_kpi
"""

import torch
import numpy as np

from imitation.data import rollout, serialize
from stable_baselines3.common.vec_env import DummyVecEnv

from utils.utils import seed_everything
from utils.gym_utils import (
    create_unified_flexiblearmenv_and_controller_and_safety_filter,
)

SEED = 0
rng = np.random.default_rng(SEED)
seed_everything(SEED)

DEMO_DIR = "demos"

# TODO: Hydra config for each algorithm
# TODO: With and without safety filter


env, _, _ = create_unified_flexiblearmenv_and_controller_and_safety_filter(
    create_controller=False, add_wall_obstacle=True, create_safety_filter=False
)
venv = DummyVecEnv([lambda: env])

# --- BC ---
bc_model = torch.load("trained_models/IL/BC/2023-09-04_14-44/SEED_0/best_model.pt")
bc_rollouts = rollout.rollout(
    policy=bc_model,
    venv=venv,
    sample_until=rollout.make_sample_until(min_episodes=10),
    rng=rng,
    unwrap=False,
    exclude_infos=True,
    verbose=True,
    render=True,
)
serialize.save(f"{DEMO_DIR}/bc.pkl", bc_rollouts)
# --------------------------

# --- DAGGER ---
dagger_model = torch.load(
    "trained_models/IL/DAGGER/2023-09-04_14-44/SEED_0/best_model.pt"
)
dagger_rollouts = rollout.rollout(
    policy=dagger_model,
    venv=venv,
    sample_until=rollout.make_sample_until(min_episodes=10),
    rng=rng,
    unwrap=False,
    exclude_infos=True,
    verbose=True,
    render=True,
)
serialize.save(f"{DEMO_DIR}/dagger.pkl", dagger_rollouts)
# --------------------------
