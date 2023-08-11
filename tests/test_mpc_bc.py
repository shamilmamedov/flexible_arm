"""
This demo loads MPC rollouts as the expert and uses BC for imitation learning.
RUN COMMAND: python -m tests.test_mpc_bc
"""

import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data import serialize

from envs.gym_env import FlexibleArmEnv, FlexibleArmEnvOptions
from utils.utils import StateType


rng = np.random.default_rng(0)

# --- Create FlexibleArm environment ---
n_seg = 5
n_seg_mpc = 3

# Create initial/final state: (base-rotation, base-bend, elbow-bend)
qa_initial = np.array([np.pi / 2, np.pi / 10, -np.pi / 8])
qa_final = np.array([0.0, 2 * np.pi / 5, -np.pi / 3])

# create data environment
R_Q = [3e-6] * 3
R_DQ = [2e-3] * 3
R_PEE = [1e-4] * 3
env_options = FlexibleArmEnvOptions(
    n_seg=n_seg,
    n_seg_estimator=n_seg_mpc,
    sim_time=1.3,
    dt=0.01,
    qa_start=qa_initial,
    qa_end=qa_final,
    qa_range_end=np.array([1.0, 1.0, 1.0]),
    contr_input_states=StateType.ESTIMATED,  # "real" if the n_seg is the same for the data and control env
    sim_noise_R=np.diag([*R_Q, *R_DQ, *R_PEE]),
    render_mode="human",
)
env = FlexibleArmEnv(env_options)
# --------------------------------------

rollouts = serialize.load("mpc_expert_rollouts.pkl")

transitions = rollout.flatten_trajectories(rollouts)

bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=rng,
)

reward, _ = evaluate_policy(
    bc_trainer.policy,  # type: ignore[arg-type]
    env,
    n_eval_episodes=3,
    render=False,
)
print(f"Reward before training: {reward}")

print("Training a policy using Behavior Cloning")
bc_trainer.train(n_epochs=10)

reward, _ = evaluate_policy(
    bc_trainer.policy,  # type: ignore[arg-type]
    env,
    n_eval_episodes=3,
    render=True,
)
print(f"Reward after training: {reward}")
