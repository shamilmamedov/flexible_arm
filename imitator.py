from dataclasses import dataclass
import numpy as np
from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
import tempfile
from typing import TYPE_CHECKING, Tuple
from stable_baselines3.common.env_checker import check_env

import plotting
from animation import Panda3dAnimator
from controller import NNController
from gym_env import FlexibleArmEnvOptions, FlexibleArmEnv
from gym_utils import CallableExpert
from mpc_3dof import Mpc3Dof
from flexible_arm_3dof import get_rest_configuration
from simulation import SimulatorOptions, Simulator

# Avoid circular imports with type checking
if TYPE_CHECKING:
    from flexible_arm_3dof import FlexibleArm3DOF, SymbolicFlexibleArm3DOF


@dataclass
class ImitatorOptions:
    """
    Options for imitation learning
    """

    def __init__(self, dt: float = 0.01):
        self.environment_options: FlexibleArmEnvOptions = FlexibleArmEnvOptions(dt=dt)
        self.filename: str = "trained_policy"
        self.save_file: bool = True
        self.n_episodes: int = 1000  # number of training episodes (100 ~ 1 minute on laptop)
        self.rollout_round_min_episodes: int = 2  # option for dagger algorithm.
        self.rollout_round_min_timesteps: int = 1000  # option for dagger algorithm.


class Imitator:
    """
    Imitation class, used for learning an MPC policy
    """

    def __init__(self, options: ImitatorOptions, expert_controller: Mpc3Dof, estimator=None):
        self.options = options
        self.expert_controller = expert_controller
        self.estimator = estimator

        # create and sanity check training environment
        self.env = FlexibleArmEnv(options=options.environment_options, estimator=estimator)
        check_env(self.env)

        # set mpc reference and make baselines expert
        u_ref = np.zeros((3,))
        q_mpc = get_rest_configuration(self.env.xee_final[:, 0], expert_controller.options.n_seg)
        dq_mpc = np.zeros_like(q_mpc)
        x_mpc = np.vstack((q_mpc, dq_mpc))
        self.expert_controller.set_reference_point(p_ee_ref=self.env.xee_final, x_ref=x_mpc, u_ref=u_ref)
        self.callable_expert = CallableExpert(expert_controller, observation_space=self.env.observation_space,
                                              action_space=self.env.action_space)

        self.venv = DummyVecEnv([lambda: FlexibleArmEnv(options=options.environment_options, estimator=estimator)])
        self.bc_trainer = bc.BC(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space
        )

    def evaluate_expert(self, n_eps: int = 1):
        total_reward_sum = 0
        for count_eps in range(n_eps):
            obs = self.env.reset()
            reward_sum = 0
            for step in range(1000):
                a = self.callable_expert.predict_np(obs)
                obs, reward, done, info = self.env.step(a)
                reward_sum += reward
                if done:
                    print("Final expert reward in episode {}: {}".format(count_eps, reward_sum))
                    break
            total_reward_sum += reward_sum
        print("Average expert reward: {}".format(total_reward_sum / n_eps))
        self.env.reset()

    def evaluate_student(self):
        controller = NNController(nn_file=self.options.filename, n_seg=self.expert_controller.options.n_seg)

        # simulate
        n_iter = 400
        if self.estimator is not None:
            sim_opts = SimulatorOptions(contr_input_states='estimated')
        else:
            sim_opts = SimulatorOptions()
        sim = Simulator(self.env.model_sym, controller, 'cvodes', self.estimator, opts=sim_opts)
        x0 = self.env.reset()
        x, u, y, xhat = sim.simulate(x0.flatten(), self.env.dt, n_iter)
        t = np.arange(0, n_iter + 1) * self.env.dt

        # Parse joint positions
        n_skip = 1
        q = x[::n_skip, :self.env.model_sym.nq]

        # Visualization
        plotting.plot_controls(t[:-1], u)
        plotting.plot_measurements(t, y)

        # Animate simulated motion
        animator = Panda3dAnimator(self.env.model_sym.urdf_path, self.env.dt * n_skip, q).play(3)

    def train(self):
        with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
            print(tmpdir)
            dagger_trainer = SimpleDAggerTrainer(
                venv=self.venv,
                scratch_dir=tmpdir,
                expert_policy=self.callable_expert,
                bc_trainer=self.bc_trainer,
            )
            dagger_trainer.train(self.options.n_episodes,
                                 rollout_round_min_episodes=self.options.rollout_round_min_episodes,
                                 rollout_round_min_timesteps=self.options.rollout_round_min_timesteps)

        # Evaluate and save trained policy
        reward, _ = evaluate_policy(dagger_trainer.policy, self.env, 10)
        print("Final reward: {}".format(reward))

        # Save policy
        if self.options.save_file:
            dagger_trainer.policy.save(self.options.filename)
