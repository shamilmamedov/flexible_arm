import pickle
from copy import copy
from dataclasses import dataclass, field
from typing import List, Tuple
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt

import plotting
from animation import Panda3dAnimator
from controller import NNController
from flexible_arm_3dof import get_rest_configuration, SymbolicFlexibleArm3DOF
from imitation_builder import ImitationBuilder
import numpy as np

from mpc_3dof import Mpc3Dof
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
    wall_penetrations: List = field(default_factory=lambda: [])
    speed_violation: List = field(default_factory=lambda: [])


def latexify():
    params = {
        "backend": "ps",
        "text.latex.preamble": r"\usepackage{gensymb} \usepackage{amsmath}",
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "text.usetex": True,
        "font.family": "serif",
    }

    matplotlib.rcParams.update(params)


class Evaluator:
    def __init__(self, builder: ImitationBuilder, n_episodes: int = 10, policy_dir: str = None,
                 render: bool = False, show_plots: bool = False, n_mpc: List[int] = None):
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
        self.mpc_n_kpis = []
        self.mpc_n_timings = []
        self.policy_dir = policy_dir
        self.epsilons = [0.2, 0.1, 0.05, 0.025]
        self.render = render
        self.show_plots = show_plots
        if n_mpc is None:
            n_mpc = []
        self.n_mpc = n_mpc

    def evaluate_all(self, policy_dir, save_file_name: str = 'expert_eval.txt', seed: int = 1):
        self.evaluate_expert(seed=seed)
        self.evaluate_nn(policy_dir=policy_dir, seed=seed)
        self.evaluate_nn_safe(policy_dir=policy_dir, seed=seed)
        for n in self.n_mpc:
            self.evaluate_n_mpc(n=n, seed=seed)

        data = [self.expert_kpis, self.expert_timings,
                self.nn_kpis, self.nn_timings,
                self.nn_safety_kpi, self.nn_safety_timings,
                self.mpc_n_kpis, self.mpc_n_timings]

        dbfile = open(self.policy_dir + save_file_name, 'wb')
        pickle.dump(data, dbfile)
        dbfile.close()

    def print_all(self, save_file_name: str = 'expert_eval.txt'):
        dbfile = open(self.policy_dir + save_file_name, 'rb')
        [self.expert_kpis, self.expert_timings,
         self.nn_kpis, self.nn_timings,
         self.nn_safety_kpi, self.nn_safety_timings,
         self.mpc_n_kpis, self.mpc_n_timings] = pickle.load(dbfile)

        dbfile.close()

        self.print_result()

    def print_result(self):
        assert self.expert_kpis.__len__() > 0 and self.nn_kpis.__len__() > 0
        assert self.expert_kpis[0].epsilon == self.nn_kpis[0].epsilon

        header = ["approach",
                  "t_exec mean",
                  "t_exec std",
                  "t_exec min",
                  "t_exec max",
                  *[["fail, eps=" + str(epsilon), "t_ball", "d_ball"] for epsilon in
                    self.nn_kpis[0].epsilon],
                  "vel. violation",
                  "obst. violation"]
        header = flatten(header)
        n_col = header.__len__()
        n_row = 3 + self.n_mpc.__len__()
        acc_str = "{:3.2f}"
        percentage_str = "{:3.0f}"

        data = []
        approach_names = ["expert MPC", "NN", "NN + SF"] + ["MPC, ($N=" + str(mpc_len) + "$)" for mpc_len in self.n_mpc]
        for i, name in zip(range(n_row), approach_names):
            data.append([name] + ["empty"] * (n_col - 1))

        timings = [self.expert_timings, self.nn_timings, self.nn_safety_timings, *self.mpc_n_timings]
        for i, timings in zip(range(n_row), timings):
            data[i][3] = "& {:3.3f}".format(np.mean(np.array([timing.min for timing in timings])) * 1e3)
            data[i][4] = "& {:3.3f}".format(np.mean(np.array([timing.max for timing in timings])) * 1e3)
            data[i][1] = "& {:3.3f}".format(np.mean(np.array([timing.mean for timing in timings])) * 1e3)
            data[i][2] = "& {:3.3f}".format(np.mean(np.array([timing.std for timing in timings])) * 1e3)

        for cnt_eps, epsilon in enumerate(self.expert_kpis[0].epsilon):
            nn_path_lens = [kpi.path_len[cnt_eps] for kpi in self.nn_kpis]
            nn_safe_path_lens = [kpi.path_len[cnt_eps] for kpi in self.nn_safety_kpi]
            total_count = self.expert_kpis.__len__()
            fail_count_nn = sum(map(lambda x: x < 0., nn_path_lens))
            fail_count_nn_safe = sum(map(lambda x: x < 0., nn_safe_path_lens))

            expert_paths = []
            expert_times = []
            for i in range(total_count):
                expert_paths.append(self.expert_kpis[i].path_len[cnt_eps])
                expert_times.append(self.expert_kpis[i].t_epsilon[cnt_eps])

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
            nn_safe_paths_normalized = []
            for i in range(total_count):
                if self.nn_safety_kpi[i].path_len[cnt_eps] > 0:
                    nn_safe_paths_normalized.append(
                        self.nn_safety_kpi[i].path_len[cnt_eps] / self.expert_kpis[i].path_len[cnt_eps])
            nn_safe_t_eps_normalized = []
            for i in range(total_count):
                if self.nn_safety_kpi[i].path_len[cnt_eps] > 0:
                    nn_safe_t_eps_normalized.append(
                        self.nn_safety_kpi[i].t_epsilon[cnt_eps] / self.expert_kpis[i].t_epsilon[cnt_eps])
            nmpc_paths_normalized = []
            nmpc_t_eps_normalized = []
            fail_counts_nmpc = []
            for cnt_nmpc in range(self.n_mpc.__len__()):
                nmpc_paths = []
                path_lenghts_nmpc = [kpi.path_len[cnt_eps] for kpi in self.mpc_n_kpis[cnt_nmpc]]
                fail_counts_nmpc.append(sum(map(lambda x: x < 0., path_lenghts_nmpc)))
                for i in range(total_count):
                    if self.mpc_n_kpis[cnt_nmpc][i].path_len[cnt_eps] > 0:
                        nmpc_paths.append(
                            self.mpc_n_kpis[cnt_nmpc][i].path_len[cnt_eps] / self.expert_kpis[i].path_len[cnt_eps])
                nmpc_paths_normalized.append(nmpc_paths)
                nmpc_t_eps = []
                for i in range(total_count):
                    if self.mpc_n_kpis[cnt_nmpc][i].path_len[cnt_eps] > 0:
                        nmpc_t_eps.append(
                            self.mpc_n_kpis[cnt_nmpc][i].t_epsilon[cnt_eps] / self.expert_kpis[i].t_epsilon[cnt_eps])
                nmpc_t_eps_normalized.append(nmpc_t_eps)

            data[0][5 + 3 * cnt_eps] = "& " + percentage_str.format(0)
            data[0][7 + 3 * cnt_eps] = ("& " + acc_str + "\pm" + acc_str).format(np.mean(np.array(expert_paths)),
                                                                                 np.std(np.array(expert_paths)))
            data[0][6 + 3 * cnt_eps] = ("& " + acc_str + "\pm" + acc_str).format(np.mean(np.array(expert_times)),
                                                                                 np.std(np.array(expert_times)))

            data[1][5 + 3 * cnt_eps] = "& " + percentage_str.format(fail_count_nn / total_count * 100)
            data[1][7 + 3 * cnt_eps] = ("& " + acc_str + "\pm" + acc_str).format(np.mean(np.array(nn_paths_normalized)),
                                                                                 np.std(np.array(nn_paths_normalized)))
            data[1][6 + 3 * cnt_eps] = ("& " + acc_str + "\pm" + acc_str).format(np.mean(np.array(nn_t_eps_normalized)),
                                                                                 np.std(np.array(nn_t_eps_normalized)))
            # print("Path Times NN: {} +- {}m".format(np.mean(np.array(nn_t_eps_normalized)),
            #                                        np.std(np.array(nn_t_eps_normalized))))
            data[2][5 + 3 * cnt_eps] = "& " + percentage_str.format(fail_count_nn_safe / total_count * 100)
            data[2][7 + 3 * cnt_eps] = ("& " + acc_str + "\pm" + acc_str).format(
                np.mean(np.array(nn_safe_paths_normalized)),
                np.std(np.array(nn_safe_paths_normalized)))
            data[2][6 + 3 * cnt_eps] = ("& " + acc_str + "\pm" + acc_str).format(
                np.mean(np.array(nn_safe_t_eps_normalized)),
                np.std(np.array(nn_safe_t_eps_normalized)))
            for i in range(len(self.n_mpc)):
                data[3 + i][5 + 3 * cnt_eps] = "& " + percentage_str.format(fail_counts_nmpc[i] / total_count * 100)
                data[3 + i][7 + 3 * cnt_eps] = ("& " + acc_str + "\pm" + acc_str).format(
                    np.mean(np.array(nmpc_paths_normalized[i])),
                    np.std(np.array(nmpc_paths_normalized[i])))
                data[3 + i][6 + 3 * cnt_eps] = ("& " + acc_str + "\pm" + acc_str).format(
                    np.mean(np.array(nmpc_t_eps_normalized[i])),
                    np.std(np.array(nmpc_t_eps_normalized[i])))

        kpis = [self.expert_kpis, self.nn_kpis, self.nn_safety_kpi, *self.mpc_n_kpis]
        for i, kpi in zip(range(n_row), kpis):
            wall_max = []
            for iteration_kpi in kpi:
                if len(iteration_kpi.wall_penetrations) > 0:
                    wall_max.append(np.max(np.array(iteration_kpi.wall_penetrations)))
                else:
                    wall_max.append(0)
            data[i][-1] = "& " + acc_str.format(1000 * np.mean(np.array(wall_max)))
            speed_max = []
            for iteration_kpi in kpi:
                if len(iteration_kpi.speed_violation) > 0:
                    speed_max.append(np.max(np.array(iteration_kpi.speed_violation)))
                else:
                    speed_max.append(0)
            data[i][-2] = "& " + acc_str.format(np.max(np.array(speed_max)))

        print(tabulate(data, headers=header))

    def post_evaluation(self, controller, timing_container, kpi_container):

        x, u, y, xhat = self.env.simulator.x, self.env.simulator.u, self.env.simulator.y, self.env.simulator.x_hat
        t = np.arange(0, self.env.simulator.opts.n_iter + 1) * self.env.dt

        n_iter, _ = y.shape
        wall_penetrations = []
        for i in range(0, n_iter):
            if y[i, 1 + 6] > 0:
                wall_penetrations.append(y[i, 1 + 6])
        speed_violation = []
        nq = self.env.simulator.robot.nq
        qa_idx = self.env.simulator.robot.qa_idx
        for i in range(0, n_iter):
            pos_volation = np.min(self.env.simulator.robot.dqa_max - x[i, nq + np.array(qa_idx)])
            if pos_volation < 0:
                speed_violation.append(-pos_volation)
            neg_volation = np.max(-self.env.simulator.robot.dqa_max - x[i, nq + np.array(qa_idx)])
            if neg_volation > 0:
                speed_violation.append(neg_volation)

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
            t_epsilons.append(ns2g * self.env.dt)
            pls.append(pl)

        kpi_container.append(Kpi(path_len=pls, t_epsilon=t_epsilons,
                                 epsilon=self.epsilons, speed_violation=speed_violation,
                                 wall_penetrations=wall_penetrations))

        if self.render:
            animator = Panda3dAnimator(self.env.model_sym.urdf_path, self.env.dt, q).play(1)
        return x, u, y, xhat, t

    def evaluate_expert(self, seed: int = 1, show_plots: bool = False):
        # simulate
        nq = int(self.expert_controller.options.q_diag.shape[0] / 2)
        dt = self.imitator.options.environment_options.dt
        for simulation_count in range(self.n_episodes):
            if seed is not None:
                np.random.seed(seed + simulation_count)
            state = self.env.reset()
            qk, dqk = np.expand_dims(state[0:nq], 1), np.expand_dims(state[nq:], 1)
            qa = qk[[0, 1, 1 + (qk.__len__() - 1) // 2]]
            self.expert_controller.reset()
            q_mpc = get_rest_configuration(qa, self.expert_controller.options.n_seg)
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

            ret_vals = self.post_evaluation(controller=self.expert_controller, timing_container=self.expert_timings,
                                            kpi_container=self.expert_kpis)
        return ret_vals

    def evaluate_n_mpc(self, n: int, seed: int = 1):
        # simulate
        nq = int(self.expert_controller.options.q_diag.shape[0] / 2)
        dt = self.imitator.options.environment_options.dt
        fa_sym_ld = SymbolicFlexibleArm3DOF(self.expert_controller.options.n_seg)
        options = self.expert_controller.options
        options.n = n
        options.tf = dt * (n - 1)
        controller = Mpc3Dof(model=fa_sym_ld, options=options)
        self.mpc_n_timings.append([])
        self.mpc_n_kpis.append([])
        for simulation_count in range(self.n_episodes):
            if seed is not None:
                np.random.seed(seed + simulation_count)
            state = self.env.reset()
            qk, dqk = np.expand_dims(state[0:nq], 1), np.expand_dims(state[nq:], 1)
            controller.reset()
            qa = qk[[0, 1, 1 + (qk.__len__() - 1) // 2]]
            q_mpc = get_rest_configuration(qa, controller.options.n_seg)
            dq_mpc = np.zeros_like(q_mpc)
            x_mpc = np.vstack((q_mpc, dq_mpc))
            controller.set_reference_point(x_ref=x_mpc,
                                           p_ee_ref=self.env.xee_final,
                                           u_ref=np.array([0, 0, 0]))
            for i in range(self.env.max_intg_steps):
                a = controller.compute_torques(q=qk, dq=dqk, t=simulation_count * dt)
                state, reward, done, info = self.env.step(a)
                qk, dqk = np.expand_dims(state[0:nq], 1), np.expand_dims(state[nq:], 1)
                if done:
                    break

            ret_vals = self.post_evaluation(controller=controller, timing_container=self.mpc_n_timings[-1],
                                            kpi_container=self.mpc_n_kpis[-1])
        return ret_vals

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
            ret_vals = self.post_evaluation(controller=controller, timing_container=self.nn_timings,
                                            kpi_container=self.nn_kpis)
        return ret_vals

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
            ret_vals = self.post_evaluation(controller=controller, timing_container=self.nn_safety_timings,
                                            kpi_container=self.nn_safety_kpi)
        return ret_vals

    def plot_eval_run(self, policy_dir: str = None, seed: int = 3, load_last_run: bool = False):
        latexify()
        linewidth=1
        if not load_last_run:
            x_exp, u_exp, y_exp, xhat_exp, t_exp = self.evaluate_expert(seed=seed)
            x_nn, u_nn, y_nn, xhat_nn, t_nn = self.evaluate_nn(policy_dir=policy_dir, seed=seed)
            x_snn, u_snn, y_snn, xhat_snn, t_snn = self.evaluate_nn_safe(policy_dir=policy_dir, seed=seed)

            all_data = [x_exp, u_exp, y_exp, xhat_exp, t_exp, x_nn, u_nn, y_nn, xhat_nn, t_nn, x_snn, u_snn, y_snn,
                        xhat_snn, t_snn]
            with open('evaluation_time_data.txt', 'wb') as f:
                pickle.dump(all_data, f)
        else:
            with open('evaluation_time_data.txt', 'rb') as f:
                all_data = pickle.load(f)
        [x_exp, u_exp, y_exp, xhat_exp, t_exp, x_nn, u_nn, y_nn, xhat_nn, t_nn, x_snn, u_snn, y_snn, xhat_snn,
         t_snn] = copy(all_data)
        N = 150
        x_exp, u_exp, y_exp, xhat_exp, t_exp = x_exp[:N, ], u_exp[:N, :], y_exp[:N, :], xhat_exp[:N, :], t_exp[:N]
        x_nn, u_nn, y_nn, xhat_nn, t_nn = x_nn[:N, :], u_nn[:N, :], y_nn[:N, :], xhat_nn[:N, :], t_nn[:N]
        x_snn, u_snn, y_snn, xhat_snn, t_snn = x_snn[:N, :], u_snn[:N, :], y_snn[:N, :], xhat_snn[:N, :], t_snn[:N]

        fig1, axs = plt.subplots(3, 3, sharex=True, figsize=(6, 3))
        axs_q = axs[:, 0]
        axs_dq = axs[:, 1]
        axs_pee = axs[:, 2]
        # fig2, axs_dq = plt.subplots(3, 1, sharex=True, figsize=(4, 6))
        # fig3, axs_pee = plt.subplots(3, 1, sharex=True, figsize=(4, 6))
        labels = ["MPC expert", "NN", "NN+SF"]
        colors = sns.color_palette("hls", 3)
        lines1 = plotting.plot_measurements_latex(t_exp, y_exp, self.env.xee_final, axs_q=axs_q, axs_dq=axs_dq, axs_pee=axs_pee,
                                         label=labels[0], b_dq=[2.5, 3.5, 3.5], delta_dq=0.5, b_zy=0, delta_zy=0.03,
                                         color=colors[0])
        lines2 = plotting.plot_measurements_latex(t_nn, y_nn, self.env.xee_final, axs_q=axs_q, axs_dq=axs_dq, axs_pee=axs_pee,
                                         label=labels[1], b_dq=[2.5, 3.5, 3.5], delta_dq=0.5, b_zy=0, delta_zy=0.03,
                                         color=colors[1])
        lines3 = plotting.plot_measurements_latex(t_snn, y_snn, self.env.xee_final, axs_q=axs_q, axs_dq=axs_dq, axs_pee=axs_pee,
                                         label=labels[2], b_dq=[2.5, 3.5, 3.5], delta_dq=0.5, b_zy=0, delta_zy=0.03,
                                         color=colors[2])
        lines = [lines1[0], lines2[0], lines3[0], lines1[1], lines1[-2], lines1[-1]]
        fig1.legend(lines, labels+["reference", "constraint", r'robust $\delta_{\{\dot{q},z}\}$'], ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.15))
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        #plt.show()
        fig4, axs_u = plt.subplots(3, 1, sharex=True, figsize=(4, 6))
        for i in range(3):
            axs_u[i].plot(t_exp, u_exp[:, i], label=labels[0])
            axs_u[i].plot(t_exp, u_nn[:, i], label=labels[1])
            axs_u[i].plot(t_exp, u_snn[:, i], label=labels[2])
            axs_u[i].grid(alpha=0.5)
            axs_u[i].set_ylabel(r"$\tau_{" + str(i) + r"}$")
        fig1.tight_layout()
        fig1.savefig("fig1" + ".pdf", dpi=300, bbox_inches='tight')
        # fig2.tight_layout()
        # fig2.savefig("fig2" + ".pdf", dpi=300)
        # fig3.tight_layout()
        # fig3.savefig("fig3" + ".pdf", dpi=300)
        fig4.tight_layout()
        # fig4.savefig("fig4" + ".pdf", dpi=300)
