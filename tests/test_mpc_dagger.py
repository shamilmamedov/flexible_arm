"""
This demo loads MPC rollouts as the expert and uses DAGGER for imitation learning.
RUN COMMAND: python -m tests.test_mpc_dagger
"""
import logging
import tempfile
import os

import torch
import numpy as np

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.data import serialize

from envs.gym_env import FlexibleArmEnv, FlexibleArmEnvOptions, SymbolicFlexibleArm3DOF
from mpc_3dof import Mpc3dofOptions, Mpc3Dof
from utils.gym_utils import CallableMPCExpert
from utils.utils import StateType, seed_everything

logging.basicConfig(level=logging.INFO)
TRAIN_MODEL = False
SEED = 0
rng = np.random.default_rng(SEED)
seed_everything(SEED)


# --- Create FlexibleArm environment ---
n_seg = 5
n_seg_mpc = 3

# create data environment
R_Q = [3e-6] * 3
R_DQ = [2e-3] * 3
R_PEE = [1e-4] * 3
env_options = FlexibleArmEnvOptions(
    n_seg=n_seg,
    n_seg_estimator=n_seg_mpc,
    sim_time=1.3,
    dt=0.01,
    qa_range_start=np.array([np.pi // 6, np.pi // 6, np.pi // 6]),
    qa_range_end=np.array([np.pi // 2, np.pi // 2, np.pi // 2]),
    contr_input_states=StateType.ESTIMATED,  # "real" if the n_seg is the same for the data and control env
    sim_noise_R=np.diag([*R_Q, *R_DQ, *R_PEE]),
    render_mode="human",
)
env = FlexibleArmEnv(env_options)
venv = DummyVecEnv([lambda: env])
# --------------------------------------

if TRAIN_MODEL:
    logging.info("Training a DAGGER model")
    # --- Create MPC controller ---
    fa_sym_mpc = SymbolicFlexibleArm3DOF(n_seg_mpc)
    mpc_options = Mpc3dofOptions(n_seg=n_seg_mpc, tf=1.3, n=130)
    controller = Mpc3Dof(model=fa_sym_mpc, x0=None, pee_0=None, options=mpc_options)

    # create MPC expert
    expert = CallableMPCExpert(
        controller,
        observation_space=env.observation_space,
        action_space=env.action_space,
        observation_includes_goal=True,
    )
    # -----------------------------

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        rng=rng,
    )

    rollouts = serialize.load("mpc_expert_rollouts.pkl")

    with tempfile.TemporaryDirectory(prefix="dagger_trained_") as tmpdir:
        print(tmpdir)
        dagger_trainer = SimpleDAggerTrainer(
            venv=venv,
            scratch_dir=tmpdir,
            expert_policy=expert,
            bc_trainer=bc_trainer,
            rng=rng,
            expert_trajs=rollouts,
        )

        env.reset(seed=SEED)
        reward, _ = evaluate_policy(
            dagger_trainer.policy,  # type: ignore[arg-type]
            env,
            n_eval_episodes=3,
            render=False,
        )
        print(f"Reward before training: {reward}")

        print("Training a policy using Dagger")
        dagger_trainer.train(100)

        policy = dagger_trainer.policy
        os.makedirs("trained_models", exist_ok=True)
        torch.save(policy, "trained_models/policy_mpc_dagger.pt")

else:
    logging.info("Loading a trained DAGGER model")
    policy = torch.load("trained_models/policy_mpc_dagger.pt")

env.reset(seed=SEED)
reward, _ = evaluate_policy(
    policy,  # type: ignore[arg-type]
    env,
    n_eval_episodes=3,
    render=True,
)
print(f"Reward after training: {reward}")
