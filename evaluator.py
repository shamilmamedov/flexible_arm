from dataclasses import dataclass, field
from typing import List, Tuple

import plotting
from animation import Panda3dAnimator
from controller import NNController
from flexible_arm_3dof import get_rest_configuration
from imitation_builder import ImitationBuilder
import numpy as np

from safty_filter_3dof import get_safe_controller_class
from utils import print_timings
import kpi
from tabulate import tabulate
from collections import Iterable


def flatten(A):
    rt = []
    for i in A:
        if isinstance(i, list):
            rt.extend(flatten(i))
        else:
            rt.append(i)
    return rt


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
    def __init__(self, builder: ImitationBuilder, n_episodes: int = 10, policy_dir: str = None,
                 render: bool = False, show_plots: bool = False):
        self.imitator, self.env, self.expert_controller, self.safety_filter = builder.build()
        self.env.stop_if_goal_condition = False
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
        self.render = render
        self.show_plots = show_plots

    def print_result(self):
        assert self.expert_kpis.__len__() > 0 and self.nn_kpis.__len__() > 0
        assert self.expert_kpis[0].epsilon == self.nn_kpis[0].epsilon

        header = ["Approach",
                  "t_exec mean",
                  "t_exec std",
                  "t_exec min",
                  "t_exec max",
                  *[["fail, eps=" + str(epsilon), "t_ball", "d_ball"] for epsilon in
                    self.nn_kpis[0].epsilon],
                  "vel. violation",
                  "obst. violation"]
        header = flatten(header)
        data_mpc, data_nn, data_nns = ["empty"] * header.__len__(), ["empty"] * header.__len__(), [
            "empty"] * header.__len__()
        data_mpc[0] = "MPC"
        data_nn[0] = "NN"
        data_nns[0] = "safe NN"

        data_mpc[3] = "{:3.3f}".format(np.mean(np.array([timing.min for timing in self.expert_timings])) * 1e3)
        data_mpc[4] = "{:3.3f}".format(np.mean(np.array([timing.max for timing in self.expert_timings])) * 1e3)
        data_mpc[1] = "{:3.3f}".format(np.mean(np.array([timing.mean for timing in self.expert_timings])) * 1e3)
        data_mpc[2] = "{:3.3f}".format(np.mean(np.array([timing.std for timing in self.expert_timings])) * 1e3)

        data_nn[3] = "{:3.3f}".format(np.mean(np.array([timing.min for timing in self.nn_timings])) * 1e3)
        data_nn[4] = "{:3.3f}".format(np.mean(np.array([timing.max for timing in self.nn_timings])) * 1e3)
        data_nn[1] = "{:3.3f}".format(np.mean(np.array([timing.mean for timing in self.nn_timings])) * 1e3)
        data_nn[2] = "{:3.3f}".format(np.mean(np.array([timing.std for timing in self.nn_timings])) * 1e3)

        data_nns[3] = "{:3.3f}".format(np.mean(np.array([timing.min for timing in self.nn_safety_timings])) * 1e3)
        data_nns[4] = "{:3.3f}".format(np.mean(np.array([timing.max for timing in self.nn_safety_timings])) * 1e3)
        data_nns[1] = "{:3.3f}".format(np.mean(np.array([timing.mean for timing in self.nn_safety_timings])) * 1e3)
        data_nns[2] = "{:3.3f}".format(np.mean(np.array([timing.std for timing in self.nn_safety_timings])) * 1e3)

        for cnt_eps, epsilon in enumerate(self.expert_kpis[0].epsilon):
            nn_path_lens = [kpi.path_len[cnt_eps] for kpi in self.nn_kpis]
            nn_safe_path_lens = [kpi.path_len[cnt_eps] for kpi in self.nn_safety_kpi]
            total_count = self.expert_kpis.__len__()
            fail_count_nn = sum(map(lambda x: x < 0., nn_path_lens))
            fail_count_nn_safe = sum(map(lambda x: x < 0., nn_safe_path_lens))
            nn_paths_normalized = []
            expert_paths = []
            expert_times = []
            for i in range(total_count):
                expert_paths.append(self.expert_kpis[i].path_len[cnt_eps])
                expert_times.append(self.expert_kpis[i].t_epsilon[cnt_eps])
            for i in range(total_count):
                if self.nn_kpis[i].path_len[cnt_eps] > 0:
                    nn_paths_normalized.append(
                        self.nn_kpis[i].path_len[cnt_eps] / self.expert_kpis[i].path_len[cnt_eps])
            nn_safe_paths_normalized = []
            for i in range(total_count):
                if self.nn_safety_kpi[i].path_len[cnt_eps] > 0:
                    nn_safe_paths_normalized.append(
                        self.nn_safety_kpi[i].path_len[cnt_eps] / self.expert_kpis[i].path_len[cnt_eps])
            nn_t_eps_normalized = []
            for i in range(total_count):
                if self.nn_kpis[i].path_len[cnt_eps] > 0:
                    nn_t_eps_normalized.append(
                        self.nn_kpis[i].t_epsilon[cnt_eps] / self.expert_kpis[i].t_epsilon[cnt_eps])
            nn_safe_t_eps_normalized = []
            for i in range(total_count):
                if self.nn_safety_kpi[i].path_len[cnt_eps] > 0:
                    nn_safe_t_eps_normalized.append(
                        self.nn_safety_kpi[i].t_epsilon[cnt_eps] / self.expert_kpis[i].t_epsilon[cnt_eps])

            acc_str = "{:4.4f}"
            data_mpc[5 + 3 * cnt_eps] = 0.
            data_mpc[7 + 3 * cnt_eps] = (acc_str + "+-" + acc_str).format(np.mean(np.array(expert_paths)),
                                                                          np.std(np.array(expert_paths)))
            data_mpc[6 + 3 * cnt_eps] = (acc_str + "+-" + acc_str).format(np.mean(np.array(expert_times)),
                                                                          np.std(np.array(expert_times)))

            data_nn[5 + 3 * cnt_eps] = fail_count_nn / total_count * 100
            data_nn[7 + 3 * cnt_eps] = (acc_str + "+-" + acc_str).format(np.mean(np.array(nn_paths_normalized)),
                                                                         np.std(np.array(nn_paths_normalized)))
            data_nn[6 + 3 * cnt_eps] = (acc_str + "+-" + acc_str).format(np.mean(np.array(nn_t_eps_normalized)),
                                                                         np.std(np.array(nn_t_eps_normalized)))
            # print("Path Times NN: {} +- {}m".format(np.mean(np.array(nn_t_eps_normalized)),
            #                                        np.std(np.array(nn_t_eps_normalized))))
            data_nns[5 + 3 * cnt_eps] = fail_count_nn_safe / total_count * 100
            data_nns[7 + 3 * cnt_eps] = (acc_str + "+-" + acc_str).format(np.mean(np.array(nn_safe_paths_normalized)),
                                                                          np.std(np.array(nn_safe_paths_normalized)))
            data_nns[6 + 3 * cnt_eps] = (acc_str + "+-" + acc_str).format(np.mean(np.array(nn_safe_t_eps_normalized)),
                                                                          np.std(np.array(nn_safe_t_eps_normalized)))

        print(tabulate([data_mpc, data_nn, data_nns], headers=header))

    def post_evaluation(self, controller, timing_container, kpi_container):

        x, u, y, xhat = self.env.simulator.x, self.env.simulator.u, self.env.simulator.y, self.env.simulator.x_hat
        t = np.arange(0, self.env.simulator.opts.n_iter + 1) * self.env.dt

        # Visualization
        if self.show_plots:
            plotting.plot_measurements(t, y, self.env.xee_final)

        t_mean, t_std, t_min, t_max = controller.get_timing_statistics()
        timing_container.append(Timings(min=t_min, max=t_max, std=t_std, mean=t_mean))
        # print_timings(t_mean, t_std, t_min, t_max)

        # Compute KPIs
        q = x[:, :self.env.model.nq]
        t_epsilons = []
        pls = []
        for epsilon in self.epsilons:
            ns2g = kpi.execution_time(q, self.env.simulator.robot, self.env.xee_final, epsilon)
            pl = -1
            if ns2g >= 0:
                q_kpi = q[:ns2g, :]
                pl = kpi.path_length(q_kpi, self.env.simulator.robot)
            t_epsilons.append(ns2g)
            pls.append(pl)

        kpi_container.append(Kpi(path_len=pls, t_epsilon=t_epsilons,
                                 epsilon=self.epsilons, safety_violation=0))

        if self.render:
            animator = Panda3dAnimator(self.env.model_sym.urdf_path, self.env.dt, q).play(1)

    def evaluate_expert(self, seed: int = 1, show_plots: bool = False):
        # simulate
        nq = int(self.expert_controller.options.q_diag.shape[0] / 2)
        dt = self.imitator.options.environment_options.dt
        for simulation_count in range(self.n_episodes):
            if seed is not None:
                np.random.seed(seed + simulation_count)
            state = self.env.reset()
            qk, dqk = np.expand_dims(state[0:nq], 1), np.expand_dims(state[nq:], 1)
            self.expert_controller.reset()
            q_mpc = get_rest_configuration(self.env.xee_final[:, 0], self.expert_controller.options.n_seg)
            dq_mpc = np.zeros_like(q_mpc)
            x_mpc = np.vstack((q_mpc, dq_mpc))
            self.expert_controller.set_reference_point(x_ref=x_mpc,
                                                       p_ee_ref=self.env.xee_final,
                                                       u_ref=np.array([0, 0, 0]))
            for i in range(self.env.max_intg_steps):
                a = self.expert_controller.compute_torques(q=qk, dq=dqk, t=simulation_count * dt)
                state, reward, done, info = self.env.step(a)
                qk, dqk = np.expand_dims(state[0:nq], 1), np.expand_dims(state[nq:], 1)
                if done:
                    break

            self.post_evaluation(controller=self.expert_controller, timing_container=self.expert_timings,
                                 kpi_container=self.expert_kpis)

    def evaluate_nn(self, seed: int = 1, show_plots: bool = False,
                    policy_dir: str = None, filename: str = "trained_policy"):
        if policy_dir is None:
            if self.policy_dir is None:
                raise NotImplementedError
            policy_dir = self.policy_dir
        filename = policy_dir + "/" + filename
        controller = NNController(nn_file=filename, n_seg=self.expert_controller.options.n_seg)
        nq = int(self.expert_controller.options.q_diag.shape[0] / 2)
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
            self.post_evaluation(controller=controller, timing_container=self.nn_timings,
                                 kpi_container=self.nn_kpis)

    def evaluate_nn_safe(self, seed: int = 1, show_plots: bool = False,
                         policy_dir: str = None, filename: str = "trained_policy"):
        if policy_dir is None:
            if self.policy_dir is None:
                raise NotImplementedError
            policy_dir = self.policy_dir
        filename = policy_dir + "/" + filename

        NNControllerSafe = get_safe_controller_class(NNController, safety_filter=self.safety_filter)
        controller = NNControllerSafe(nn_file=filename, n_seg=self.expert_controller.options.n_seg)

        nq = int(self.expert_controller.options.q_diag.shape[0] / 2)
        dt = self.imitator.options.environment_options.dt

        for simulation_count in range(self.n_episodes):
            if seed is not None:
                np.random.seed(seed + simulation_count)
            state = self.env.reset()
            qk, dqk = np.expand_dims(state[0:nq], 1), np.expand_dims(state[nq:], 1)
            for i in range(self.env.max_intg_steps):
                a = controller.compute_torques(q=qk, dq=dqk, t=simulation_count * dt, y=self.env.simulator.y[i, :])
                state, reward, done, info = self.env.step(a)
                qk, dqk = np.expand_dims(state[0:nq], 1), np.expand_dims(state[nq:], 1)
                if done:
                    break
            self.post_evaluation(controller=controller, timing_container=self.nn_safety_timings,
                                 kpi_container=self.nn_safety_kpi)
