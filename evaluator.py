from dataclasses import dataclass, field
from typing import List, Tuple

import plotting
from animation import Panda3dAnimator
from controller import NNController
from imitation_builder import ImitationBuilder
import numpy as np
from utils import print_timings
import kpi


@dataclass
class Timings:
    min: float = None
    max: float = None
    mean: float = None
    std: float = None


@dataclass
class Kpi:
    path_len: List = field(default_factory=lambda: [])
    t_epsilon: List = field(default_factory=lambda: [])
    epsilon: List = field(default_factory=lambda: [])
    safety_violation: int = 0


class Evaluator:
    def __init__(self, builder: ImitationBuilder, n_episodes: int = 10, policy_dir: str = None):
        self.imitator, self.env, self.controller = builder.build()
        self.n_episodes = n_episodes
        self.animate = False
        self.expert_timings = []
        self.nn_timings = []
        self.nn_safety_timings = []
        self.expert_kpis = []
        self.nn_kpis = []
        self.nn_safety_kpi = []
        self.policy_dir = policy_dir
        self.epsilons = [0.2, 0.1, 0.05, 0.025]

    def print_result(self):
        assert self.expert_kpis.__len__() > 0 and self.nn_kpis.__len__() > 0
        assert self.expert_kpis[0].epsilon == self.nn_kpis[0].epsilon

        for cnt_eps, epsilon in enumerate(self.expert_kpis[0].epsilon):
            expert_path_lens = [kpi.path_len[cnt_eps] for kpi in self.expert_kpis]
            nn_path_lens = [kpi.path_len[cnt_eps] for kpi in self.nn_kpis]
            total_count = self.expert_kpis.__len__()
            fail_count = sum(map(lambda x: x < 0., nn_path_lens))
            nn_paths_normalized = []
            for i in range(total_count):
                if self.nn_kpis[i].path_len[cnt_eps] > 0:
                    nn_paths_normalized.append(
                        self.nn_kpis[i].path_len[cnt_eps] / self.expert_kpis[i].path_len[cnt_eps])
            nn_t_eps_normalized = []
            for i in range(total_count):
                if self.nn_kpis[i].path_len[cnt_eps] > 0:
                    nn_t_eps_normalized.append(
                        self.nn_kpis[i].t_epsilon[cnt_eps] / self.expert_kpis[i].t_epsilon[cnt_eps])
            print("---------------------------------------")
            print("Epsilon: {}m".format(epsilon))
            print("Failed: {}%".format(fail_count / total_count * 100))
            print("Path Lenghts NN: {}m".format(nn_paths_normalized))
            print("Path Times NN: {}s".format(nn_t_eps_normalized))

    def evaluate_expert(self, seed: int = 1, show_plots: bool = False):
        # simulate
        nq = int(self.controller.options.q_diag.shape[0] / 2)
        dt = self.imitator.options.environment_options.dt
        for simulation_count in range(self.n_episodes):
            if seed is not None:
                np.random.seed(seed + simulation_count)
            state = self.env.reset()
            qk, dqk = np.expand_dims(state[0:nq], 1), np.expand_dims(state[nq:], 1)
            self.controller.reset()
            self.controller.set_reference_point(x_ref=self.env.x_final,
                                                p_ee_ref=self.env.xee_final,
                                                u_ref=np.array([0, 0, 0]))
            for i in range(self.env.max_intg_steps):
                a = self.controller.compute_torques(q=qk, dq=dqk, t=simulation_count * dt)
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
            if show_plots:
                plotting.plot_controls(t[:-1], u)
                plotting.plot_measurements(t, y)

            t_mean, t_std, t_min, t_max = self.controller.get_timing_statistics()
            self.expert_timings.append(Timings(min=t_min, max=t_max, std=t_std, mean=t_mean))
            # print_timings(t_mean, t_std, t_min, t_max)

            # Compute KPIs
            q = x[:, :self.env.model.nq]
            t_epsilons = []
            pls = []
            for epsilon in self.epsilons:
                ns2g = kpi.execution_time(q, self.env.simulator.robot, self.env.xee_final, epsilon)
                pl = -1
                if ns2g > 0:
                    q_kpi = q[:ns2g, :]
                    pl = kpi.path_length(q_kpi, self.env.simulator.robot)
                t_epsilons.append(ns2g)
                pls.append(pl)

            self.expert_kpis.append(Kpi(path_len=pls, t_epsilon=t_epsilons,
                                        epsilon=self.epsilons, safety_violation=0))

            # Animate simulated motion
            if self.animate:
                animator = Panda3dAnimator(self.env.model_sym.urdf_path, self.env.dt * n_skip, q).play(1)

    def evaluate_nn(self, seed: int = 1, show_plots: bool = False,
                    policy_dir: str = None, filename: str = "trained_policy"):
        if policy_dir is None:
            if self.policy_dir is None:
                raise NotImplementedError
            policy_dir = self.policy_dir
        filename = policy_dir + "/" + filename
        controller = NNController(nn_file=filename, n_seg=self.controller.options.n_seg)
        nq = int(self.controller.options.q_diag.shape[0] / 2)
        dt = self.imitator.options.environment_options.dt

        for simulation_count in range(self.n_episodes):
            if seed is not None:
                np.random.seed(seed + simulation_count)
            state = self.env.reset()
            qk, dqk = np.expand_dims(state[0:nq], 1), np.expand_dims(state[nq:], 1)
            for i in range(self.env.max_intg_steps):
                a = controller.compute_torques(q=qk, dq=dqk, t=simulation_count * dt)
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
            if show_plots:
                plotting.plot_controls(t[:-1], u)
                plotting.plot_measurements(t, y)

            t_mean, t_std, t_min, t_max = self.controller.get_timing_statistics()
            self.nn_timings.append(Timings(min=t_min, max=t_max, std=t_std, mean=t_mean))
            # print_timings(t_mean, t_std, t_min, t_max)

            # Compute KPIs
            q = x[:, :self.env.model.nq]
            t_epsilons = []
            pls = []
            for epsilon in self.epsilons:
                ns2g = kpi.execution_time(q, self.env.simulator.robot, self.env.xee_final, epsilon)
                pl = -1
                if ns2g > 0:
                    q_kpi = q[:ns2g, :]
                    pl = kpi.path_length(q_kpi, self.env.simulator.robot)
                t_epsilons.append(ns2g)
                pls.append(pl)

            self.nn_kpis.append(Kpi(path_len=pls, t_epsilon=t_epsilons,
                                    epsilon=self.epsilons, safety_violation=0))

            # Animate simulated motion
            if self.animate:
                animator = Panda3dAnimator(self.env.model_sym.urdf_path, self.env.dt * n_skip, q).play(1)
