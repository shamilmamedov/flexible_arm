"""
This demo loads MPC rollouts as the expert and uses GAIL for imitation learning.
RUN COMMAND: python -m tests.test_mpc_gail
"""
import tempfile
import numpy as np

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.util.networks import RunningNorm
from imitation.data import serialize

from envs.gym_env import FlexibleArmEnv, FlexibleArmEnvOptions, SymbolicFlexibleArm3DOF
from mpc_3dof import Mpc3dofOptions, Mpc3Dof
from utils.gym_utils import CallableMPCExpert
from utils.utils import StateType

SEED = 0
rng = np.random.default_rng(SEED)


# --- Create FlexibleArm environment ---
n_seg = 5
n_seg_mpc = 3
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

# --- Create MPC controller ---
fa_sym_mpc = SymbolicFlexibleArm3DOF(n_seg_mpc)
mpc_options = Mpc3dofOptions(n_seg=n_seg_mpc, tf=1.3, n=130)
controller = Mpc3Dof(model=fa_sym_mpc, x0=None, pee_0=None, options=mpc_options)

expert = CallableMPCExpert(
    controller,
    observation_space=env.observation_space,
    action_space=env.action_space,
    observation_includes_goal=True,
)
# -----------------------------

# --- load expert rollouts ---
rollouts = serialize.load("mpc_expert_rollouts.pkl")
# ----------------------------

learner = PPO(
    env=env,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.00001,
    n_epochs=1,
    seed=SEED,
)
reward_net = BasicShapedRewardNet(
    observation_space=env.observation_space,
    action_space=env.action_space,
    normalize_input_layer=RunningNorm,
)
gail_trainer = GAIL(
    demonstrations=rollouts,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=2048,
    n_disc_updates_per_round=4,
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
)


# evaluate the learner before training
env.reset(seed=SEED)
learner_rewards_before_training, _ = evaluate_policy(
    model=learner,
    env=env,
    n_eval_episodes=3,
    return_episode_rewards=True,
    render=False,
)
print("mean reward before training:", np.mean(learner_rewards_before_training))


# train the learner and evaluate again
gail_trainer.train(20000)

# evaluate the learner after training
env.reset(seed=SEED)
learner_rewards_after_training, _ = evaluate_policy(
    model=learner,
    env=env,
    n_eval_episodes=3,
    return_episode_rewards=True,
    render=True,
)

print("mean reward after training:", np.mean(learner_rewards_after_training))
