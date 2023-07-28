"""This is a simple example demonstrating how to clone the behavior of an expert.

Refer to the jupyter notebooks for more detailed examples of how to use the algorithms.

RUN COMMAND: python -m tests.test_imitationlib
"""
from copy import deepcopy

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper

from envs.flexible_arm_3dof import FlexibleArm3DOF
from envs.gym_env import FlexibleArmEnv, FlexibleArmEnvOptions

# Create Cartpole environment
# env = gym.make("CartPole-v1")

# Create FlexibleArm environment
n_seg = 3
fa_ld = FlexibleArm3DOF(n_seg)

# Create initial state simulated system
angle0_0 = 0.0
angle0_1 = 0
angle0_2 = np.pi / 20

q0 = np.zeros((fa_ld.nq, 1))
q0[0] += angle0_0
q0[1] += angle0_1
q0[1 + n_seg + 1] += angle0_2

# Compute reference
q_ref = deepcopy(q0)
q_ref[0] += np.pi / 2
q_ref[1] += 0
q_ref[1 + n_seg + 1] += -np.pi / 20

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

rng = np.random.default_rng(0)


def train_expert():
    print("Training a expert.")
    expert = PPO(
        policy=MlpPolicy,
        env=env,
        seed=0,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0003,
        n_epochs=10,
        n_steps=64,
        verbose=0,
    )
    expert.learn(1000)  # Note: change this to 100000 to train a decent expert.
    return expert


def sample_expert_transitions():
    expert = train_expert()

    print("Sampling expert transitions.")
    rollouts = rollout.rollout(
        expert,
        DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
        rollout.make_sample_until(min_timesteps=None, min_episodes=50),
        rng=rng,
    )
    return rollout.flatten_trajectories(rollouts)


transitions = sample_expert_transitions()
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
