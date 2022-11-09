import pathlib
from dataclasses import dataclass
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
import tempfile
from typing import TYPE_CHECKING, Tuple
from stable_baselines3.common.logger import configure, read_csv
from stable_baselines3.common.env_checker import check_env
import plotting
from animation import Panda3dAnimator
from controller import NNController
from gym_env import FlexibleArmEnvOptions, FlexibleArmEnv
from gym_utils import CallableExpert
from mpc_3dof import Mpc3Dof
from flexible_arm_3dof import get_rest_configuration
from simulation import SimulatorOptions, Simulator
from utils import print_timings

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
        self.logdir_rel: str = "/logdata"
        self.run_name: str = None
        self.save_file: bool = True
        self.n_episodes: int = 60 * 1000  # number of training episodes (1000 ~ 1 minute on laptop)
        self.rollout_round_min_episodes: int = 5  # option for dagger algorithm.
        self.rollout_round_min_timesteps: int = 2000  # option for dagger algorithm.


class Imitator:
    """
    Imitation class, used for learning an MPC policy
    """

    def __init__(self, options: ImitatorOptions, expert_controller: Mpc3Dof, estimator=None):
        self.options = options
        self.expert_controller = expert_controller
        self.estimator = estimator

        # set up logger
        if options.run_name is None:
            run_name = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
        else:
            run_name = options.run_name
        current_dir = pathlib.Path(__file__).parent.resolve()
        path = current_dir.__str__() + self.options.logdir_rel + '/' + run_name
        self.log_dir = path
        self.custom_logger = configure(path, ["stdout", "csv", "tensorboard"])

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
            action_space=self.env.action_space,
            custom_logger=self.custom_logger
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

    def render_student(self, n_episodes: int = 1, n_replay: int = 2, show_plot: bool = True, seed: int = None,
                       policy_dir: str = None, filename: str = "trained_policy"):
        filename = policy_dir + "/" + filename
        controller = NNController(nn_file=filename, n_seg=self.expert_controller.options.n_seg)

        # simulate
        n_iter = 500
        if self.estimator is not None:
            sim_opts = SimulatorOptions(contr_input_states='estimated')
        else:
            sim_opts = SimulatorOptions()

        for simulation_count in range(n_episodes):
            if seed is not None:
                np.random.seed(seed + simulation_count)
            sim = Simulator(self.env.model_sym, controller, 'cvodes', self.estimator, opts=sim_opts)
            x0 = self.env.reset()
            x, u, y, xhat = sim.simulate(x0.flatten(), n_iter)
            t = np.arange(0, n_iter + 1) * self.env.dt

            # Parse joint positions
            n_skip = 1
            q = x[::n_skip, :self.env.model_sym.nq]

            # Visualization
            if show_plot:
                plotting.plot_controls(t[:-1], u)
                plotting.plot_measurements(t, y)

            t_mean, t_std, t_min, t_max = controller.get_timing_statistics()
            print_timings(t_mean, t_std, t_min, t_max)

            # Animate simulated motion
            animator = Panda3dAnimator(self.env.model_sym.urdf_path, self.env.dt * n_skip, q).play(n_replay)

    def render_expert(self, n_episodes: int = 1, n_replay: int = 2, show_plot: bool = True, seed: int = None):
        # simulate
        nq = int(self.expert_controller.options.q_diag.shape[0] / 2)
        dt = self.options.environment_options.dt
        for simulation_count in range(n_episodes):
            if seed is not None:
                np.random.seed(seed + simulation_count)
            state = self.env.reset()
            qk, dqk = np.expand_dims(state[0:nq], 1), np.expand_dims(state[nq:], 1)
            self.expert_controller.reset()
            self.expert_controller.set_reference_point(x_ref=self.env.x_final,
                                                       p_ee_ref=self.env.xee_final,
                                                       u_ref=np.array([0, 0, 0]))
            for i in range(self.env.max_intg_steps):
                a = self.expert_controller.compute_torques(q=qk, dq=dqk, t=simulation_count * dt)
                state, reward, done, info = self.env.step(a)
                qk, dqk = np.expand_dims(state[0:nq], 1), np.expand_dims(state[nq:], 1)
                if done:
                    break
            x, u, y, xhat = self.env.simulator.x, self.env.simulator.u, self.env.simulator.y, self.env.simulator.x_hat
            t = np.arange(0, self.env.simulator.opts.n_iter + 1) * self.env.dt

            # Parse joint positions
            n_skip = 1
            q = x[::n_skip, :self.env.model_sym.nq]

            # Visualization
            if show_plot:
                plotting.plot_controls(t[:-1], u)
                plotting.plot_measurements(t, y)

            t_mean, t_std, t_min, t_max = self.expert_controller.get_timing_statistics()
            print_timings(t_mean, t_std, t_min, t_max)

            # Animate simulated motion
            animator = Panda3dAnimator(self.env.model_sym.urdf_path, self.env.dt * n_skip, q).play(n_replay)

    def train(self):
        assert int(self.expert_controller.options.tf / self.expert_controller.options.n) == int(self.env.dt)

        with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
            print(tmpdir)
            dagger_trainer = SimpleDAggerTrainer(
                venv=self.venv,
                scratch_dir=tmpdir,
                expert_policy=self.callable_expert,
                bc_trainer=self.bc_trainer,
                custom_logger=self.custom_logger
            )

            dagger_trainer.train(self.options.n_episodes,
                                 rollout_round_min_episodes=self.options.rollout_round_min_episodes,
                                 rollout_round_min_timesteps=self.options.rollout_round_min_timesteps)

        # Evaluate and save trained policy
        reward, _ = evaluate_policy(dagger_trainer.policy, self.env, 10)
        print("Final reward: {}".format(reward))

        # Save policy
        if self.options.save_file:
            dagger_trainer.policy.save(self.log_dir + '/' + self.options.filename)


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def smoothTriangle(data, degree):
    triangle = np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1]))  # up then down
    smoothed = []

    for i in range(degree, len(data) - degree * 2):
        point = data[i:i + len(triangle)] * triangle
        smoothed.append(np.sum(point) / np.sum(triangle))
    # Handle boundaries
    smoothed = [smoothed[0]] * int(degree + degree / 2) + smoothed
    while len(smoothed) < len(data):
        smoothed.append(smoothed[-1])
    return np.array(smoothed)


def plot_training(log_dir: str, filename: str = "progress.csv", n_smooth: int = 10):
    data = read_csv(log_dir + "/" + filename)
    return_mean = data['rollout/return_mean']
    return_std = data['rollout/return_std']
    return_max = data['rollout/return_max']
    return_min = data['rollout/return_min']
    bc_loss = data['bc/loss']
    prob_true_act = data['bc/prob_true_act']

    return_mean = smoothTriangle(return_mean.array, n_smooth)
    return_std = smoothTriangle(return_std.array, n_smooth)
    return_max = smoothTriangle(return_max.array, 1)
    return_min = smoothTriangle(return_min.array, 1)
    bc_loss = smoothTriangle(bc_loss.array, 1)
    prob_true_act = smoothTriangle(prob_true_act.array, 1)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
    i = np.arange(return_mean.shape[0])
    ax1.plot(i, return_mean)
    ax1.fill_between(i, y1=return_mean + return_std, y2=return_mean - return_std, alpha=0.3)
    ax1.plot(i, return_max, color='black', alpha=0.3)
    ax1.plot(i, return_min, color='black', alpha=0.3)
    ax1.set_title('Return')

    ax2.plot(bc_loss)
    ax2.set_title('Loss')

    ax3.plot(prob_true_act)
    ax3.set_title('Probability True Action')
    plt.show()