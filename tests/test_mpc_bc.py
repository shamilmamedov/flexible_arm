"""
This demo loads MPC rollouts as the expert and uses BC for imitation learning.
RUN COMMAND: python -m tests.test_mpc_bc
"""

import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data import serialize

from envs.flexible_arm_3dof import FlexibleArm3DOF, get_rest_configuration
from envs.gym_env import FlexibleArmEnv, FlexibleArmEnvOptions


rng = np.random.default_rng(0)

# --- Create FlexibleArm environment ---
n_seg = 3
fa_ld = FlexibleArm3DOF(n_seg)

# Create initial state simulated system
qa0 = np.array([np.pi / 2, np.pi / 10, -np.pi / 8])
q0 = get_rest_configuration(qa0, n_seg)
dq0 = np.zeros_like(q0)
x0 = np.vstack((q0, dq0))
_, xee0 = fa_ld.fk_ee(q0)

# Compute reference
qa_ref = np.array([0.0, 2 * np.pi / 5, -np.pi / 3])
q_ref = get_rest_configuration(qa_ref, n_seg)
dq_ref = np.zeros_like(q_ref)
x_ref = np.vstack((q_ref, dq_ref))
_, x_ee_ref = fa_ld.fk_ee(q_ref)

# create environment
# Measurement noise covairance parameters
R_Q = [3e-6] * 3
R_DQ = [2e-3] * 3
R_PEE = [1e-4] * 3
env_options = FlexibleArmEnvOptions(
    n_seg=n_seg,
    dt=0.05,
    qa_start=q0[0:3, 0],
    qa_end=q_ref[0:3, 0],
    contr_input_states="real",
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
    render=True,
)
print(f"Reward before training: {reward}")

print("Training a policy using Behavior Cloning")
bc_trainer.train(n_epochs=1)

reward, _ = evaluate_policy(
    bc_trainer.policy,  # type: ignore[arg-type]
    env,
    n_eval_episodes=3,
    render=True,
)
print(f"Reward after training: {reward}")
