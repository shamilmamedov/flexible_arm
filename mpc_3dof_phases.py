import tempfile
from copy import copy
from dataclasses import dataclass
from tempfile import mkdtemp

import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline
from controller import BaseController
from typing import TYPE_CHECKING, Dict, Tuple
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosMultiphaseOcp, AcadosModel
from poly5_planner import initial_guess_for_active_joints, get_reference_for_all_joints
from envs.flexible_arm_3dof import (
    get_rest_configuration,
    inverse_kinematics_rb,
    compute_reference_state_and_input,
)
from utils.utils import Updatable
import casadi as ca

# Avoid circular imports with type checking
if TYPE_CHECKING:
    from envs.flexible_arm_3dof import FlexibleArm3DOF, SymbolicFlexibleArm3DOF

Q_QA = 0.01  # penalty on active joints positions # 0.1, 1
Q_QP = 0.01  # penalty on passive joints positions # 0.1, 0.001
Q_DQA = 0.1  # penalty on active joints velocities # 10., 1., 0.1,
Q_DQP = 1  # penalty on passive joints velocities # 0.001, 0.1
Q_DQA_E = 0.1  # penalty on terminal active joints velocities
Q_QA_E = 0.01  # penalty on terminal active joints velocities


def translate_config(q_in: np.ndarray, n_seg_in: int, n_seg_out: int):
    assert q_in.shape[1] == 1 + 2 * (n_seg_in + 1)
    q_out = np.zeros((1, 1 + 2 * (n_seg_out + 1)))
    q_out[0, 0] = q_in[0, 0]
    q_out[0, 1] = q_in[0, 1]
    q_out[0, 1 + n_seg_out + 1] = q_in[0, 1 + n_seg_in + 1]
    q_sum_beam_1 = np.sum(q_in[0, 2:2 + n_seg_in])
    q_sum_beam_2 = np.sum(q_in[0, 2 + n_seg_in + 1:])
    q_out[0, 2:2 + n_seg_out] = q_sum_beam_1 / n_seg_out
    q_out[0, 2 + n_seg_out + 1:] = q_sum_beam_2 / n_seg_out
    return q_out


def get_transition_model(n_in: int, n_out: int) -> AcadosModel:
    # set up states & controls
    nq_1 = n_in // 2
    n_seg_1 = int((n_in / 2 - 3) / 2)
    n_seg_2 = int((n_out / 2 - 3) / 2)
    x_in = ca.SX.sym('x_in', n_in)

    q_sum_beam_1 = ca.sum1(x_in[2:2 + n_seg_1])
    q_sum_beam_2 = ca.sum1(x_in[n_seg_1 + 2 + 1:nq_1])
    dq_sum_beam_1 = ca.sum1(x_in[nq_1 + 2:nq_1 + 2 + n_seg_1])
    dq_sum_beam_2 = ca.sum1(x_in[nq_1 + n_seg_1 + 2 + 1:])

    vec_q_beam_1_out = n_seg_2 * [q_sum_beam_1 / n_seg_2]
    vec_q_beam_2_out = n_seg_2 * [q_sum_beam_2 / n_seg_2]
    vec_dq_beam_1_out = n_seg_2 * [dq_sum_beam_1 / n_seg_2]
    vec_dq_beam_2_out = n_seg_2 * [dq_sum_beam_2 / n_seg_2]

    x_out = ca.vertcat(
        x_in[0:2],
        *vec_q_beam_1_out,
        x_in[n_seg_1 + 2],
        *vec_q_beam_2_out,
        x_in[nq_1:nq_1 + 2],
        *vec_dq_beam_1_out,
        x_in[nq_1 + n_seg_1 + 2],
        *vec_dq_beam_2_out,
    )
    # set up model
    model = AcadosModel()
    model.name = 'transition_model'
    model.x = x_in
    model.u = ca.SX.sym('u', 0, 0)
    model.disc_dyn_expr = x_out

    return model


@dataclass
class Mpc3dofPhasesOptions(Updatable):
    """
    Dataclass for MPC options
    """

    def __init__(self, n_seg_p1: int = 3, n_seg_p2: int = 1, tf: float = 2,
                 t_trans: float = 0.66666, n_trans: int = 10, n: int = 30):
        self.n_seg_p1: int = n_seg_p1  # n_seg corresponds to (1 + 2 * (n_seg + 1))*2 states
        self.n_seg_p2: int = n_seg_p2
        self.n_trans = n_trans
        self.t_trans = t_trans
        self.n: int = n  # number of discretization points
        self.tf: float = tf  # time horizon
        self.nlp_iter: int = (
            50  # number of iterations of the nonlinear solver, only used if NOT RTI
        )
        self.condensing_relative: float = 1  # relative factor of condensing [0-1]
        self.wall_constraint_on: bool = (
            True  # choose whether we activate the wall constraint
        )
        self._calcluate_derived_parameters()

    def update(self, new: Dict):
        super().update(new)
        self._calcluate_derived_parameters()

    def _calcluate_derived_parameters(self):
        # States are ordered for each link
        self.q_diag_p1: np.ndarray = np.array(
            [Q_QA] * (2)
            + [Q_QP] * (self.n_seg_p1)  # qa1 and qa2
            + [Q_QA] * (1)  # qp 1st link
            + [Q_QP] * (self.n_seg_p1)  # qa3
            + [Q_DQA] * (2)  # qp 2nd link
            + [Q_DQP] * (self.n_seg_p1)  # dqa1 and dqa2
            + [Q_DQA] * (1)  # dqp 1st link
            + [Q_DQP] * (self.n_seg_p1)  # dqa3
        )  # dqp 2nd link
        self.q_e_diag_p1: np.ndarray = np.array(
            [Q_QA_E] * (2)
            + [Q_QP] * (self.n_seg_p1)  # qa1 and qa2
            + [Q_QA_E] * (1)  # qp 1st link
            + [Q_QP] * (self.n_seg_p1)  # qa3
            + [Q_DQA_E] * (2)  # qp 2nd link
            + [Q_DQP] * (self.n_seg_p1)  # dqa1 and dqa2
            + [Q_DQA_E] * (1)  # dqp 1st link
            + [Q_DQP] * (self.n_seg_p1)  # dqa3
        )  # dqp 2nd link
        self.q_diag_p2: np.ndarray = np.array(
            [Q_QA] * (2)
            + [Q_QP] * (self.n_seg_p2)  # qa1 and qa2
            + [Q_QA] * (1)  # qp 1st link
            + [Q_QP] * (self.n_seg_p2)  # qa3
            + [Q_DQA] * (2)  # qp 2nd link
            + [Q_DQP] * (self.n_seg_p2)  # dqa1 and dqa2
            + [Q_DQA] * (1)  # dqp 1st link
            + [Q_DQP] * (self.n_seg_p2)  # dqa3
        )  # dqp 2nd link
        self.q_e_diag_p2: np.ndarray = np.array(
            [Q_QA_E] * (2)
            + [Q_QP] * (self.n_seg_p2)  # qa1 and qa2
            + [Q_QA_E] * (1)  # qp 1st link
            + [Q_QP] * (self.n_seg_p2)  # qa3
            + [Q_DQA_E] * (2)  # qp 2nd link
            + [Q_DQP] * (self.n_seg_p2)  # dqa1 and dqa2
            + [Q_DQA_E] * (1)  # dqp 1st link
            + [Q_DQP] * (self.n_seg_p2)  # dqa3
        )  # dqp 2nd link
        # weights on algebraic variables related to reference p_ee. Not needed in safety filter
        self.z_diag: np.ndarray = np.array([1] * 3) * 1e4
        self.z_e_diag: np.ndarray = np.array([1] * 3) * 1e4

        # weights on control
        self.r_diag: np.ndarray = np.array([1e0, 10e0, 10e0]) * 1e-1

        # slacks for angular speed constraints of active joints
        self.w2_slack_speed: float = 1e6
        self.w1_slack_speed: float = 1e3

        # slacks for position and wall penetration
        self.w2_slack_wall: float = 1e5
        self.w1_slack_wall: float = 1e4

        # slacks for speed limitation in wall direction
        self.w2_slack_speed_wall: float = 1e1
        self.w1_slack_speed_wall: float = 1e1

    def get_sampling_time(self) -> float:
        return self.tf / self.n


class Mpc3DofPhases(BaseController):
    """
    Controller class for 3 dof flexible link model based on acados
    """

    def __init__(
            self,
            model_p1: "SymbolicFlexibleArm3DOF",
            model_p2: "SymbolicFlexibleArm3DOF",
            x0: np.ndarray = None,
            pee_0: np.ndarray = None,
            options: Mpc3dofPhasesOptions = Mpc3dofPhasesOptions(n_seg_p1=3, n_seg_p2=1, n_trans=10, n=30, tf=2),
    ):
        """
        :parameter x0: initial state vector
        :parameter pee_0: initial end-effector position
        :parameter options: a class with options
        """
        if x0 is None:
            x0 = np.zeros((2 * (1 + 2 * (1 + options.n_seg_p1)), 1))
        if pee_0 is None:
            pee_0 = np.zeros((3, 1))
        x0_p2 = np.zeros((2 * (1 + 2 * (1 + options.n_seg_p2)), 1))
        x0_p2[0] = x0[0]
        x0_p2[1] = x0[1]
        x0_p2[1 + 1 + options.n_seg_p2] = copy(x0[1 + 1 + options.n_seg_p1])

        self.acados_tmp_dir = tempfile.mkdtemp()

        self.u_max = model_p1.tau_max  # [Nm]
        self.dq_active_max = model_p1.dqa_max  # [rad/s]

        self.fa_model_p1 = model_p1
        model_p1, constraint_expr_p1 = model_p1.get_acados_model_safety()
        self.model_p1 = model_p1

        self.fa_model_p2 = model_p2
        model_p2, constraint_expr_p2 = model_p2.get_acados_model_safety()
        self.model_p2 = model_p2

        self.options = options
        self.iteration_counter = 0
        self.inter_t2q = None
        self.inter_t2dq = None
        self.inter_pee = None
        self.p_ee_ref = None
        self.last_U = None

        # create ocp object to formulate the OCP
        multi_phase_ocp = AcadosMultiphaseOcp(N_list=[options.n_trans, 1, options.n - options.n_trans])

        # phase 0 -----------------------------------------------------------------------------------------------------
        models = [model_p1, model_p2]
        n_hor = [options.n_trans, options.n - options.n_trans]
        n_seg = [options.n_seg_p1, options.n_seg_p2]
        q_diag = [options.q_diag_p1, options.q_diag_p2]
        q_e_diag = [options.q_e_diag_p1, options.q_e_diag_p2]
        constr_expr = [constraint_expr_p1, constraint_expr_p2]
        t_hor = [options.t_trans, options.tf - options.t_trans]
        x0_phases = [x0, x0_p2]
        for phase_idx in range(len(x0_phases)):
            final_phase = False
            if phase_idx == len(x0_phases) - 1:
                final_phase = True

            ocp = AcadosOcp()
            ocp.model = models[phase_idx]  # set model

            # OCP parameter adjustment
            nx = models[phase_idx].x.size()[0]
            nu = models[phase_idx].u.size()[0]
            nz = models[phase_idx].z.size()[0]
            n_p = models[phase_idx].p.size()[0]
            ny = nx + nu + nz
            ny_e = nx + nz
            self.nu = nu
            self.nx = nx

            # some checks
            # assert nx == options.q_diag_p1.shape[0] == options.q_diag_p1.shape[0]
            # assert nu == options.r_diag.shape[0]
            # assert nz == options.z_diag.shape[0] == options.z_e_diag.shape[0]

            # ocp.model.name = (
            #         "mpc_p" + str(phase_idx) + "n_" +
            #         str(n_hor[phase_idx]) + "_seg" +
            #         str(n_seg[phase_idx])
            # )
            # ocp.code_export_directory = mkdtemp()

            # set dimensions
            ocp.dims.N = n_hor[phase_idx]

            # set cost module
            ocp.cost.cost_type = "LINEAR_LS"
            ocp.cost.cost_type_e = "LINEAR_LS"

            Q = np.diagflat(q_diag[phase_idx])
            Q_e = np.diagflat(q_e_diag[phase_idx])
            R = np.diagflat(options.r_diag)
            Z = np.diagflat(options.z_diag)
            Z_e = np.diagflat(options.z_e_diag)

            ocp.cost.W = scipy.linalg.block_diag(Q, R, Z)
            ocp.cost.W_e = scipy.linalg.block_diag(Q_e, Z_e)

            ocp.cost.Vx = np.zeros((ny, nx))
            ocp.cost.Vx[:nx, :nx] = np.eye(nx)

            Vu = np.zeros((ny, nu))
            Vu[nx: nx + nu, :] = np.eye(nu)
            ocp.cost.Vu = Vu

            Vz = np.zeros((ny, nz))
            Vz[nx + nu:, :] = np.eye(nz)
            ocp.cost.Vz = Vz

            ocp.cost.Vx_e = np.zeros((ny_e, nx))
            ocp.cost.Vx_e[:nx, :nx] = np.eye(nx)

            # Vz_e = np.zeros((ny_e, nz))
            # Vz_e[nx:, :] = np.eye(nz)
            # ocp.cost.Vz_e = Vz_e

            x_goal = x0_phases[phase_idx]
            x_goal_cartesian = pee_0  # np.expand_dims(np.array([x_cartesian, y_cartesian, z_cartesian]), 1)
            ocp.cost.yref = np.vstack(
                (x_goal, np.zeros((nu, 1)), x_goal_cartesian)
            ).flatten()
            ocp.cost.yref_e = np.vstack((x_goal, x_goal_cartesian)).flatten()

            # general constraints
            ocp.constraints.constr_type = "BGH"
            if phase_idx == 0:
                ocp.constraints.x0 = x0_phases[phase_idx].reshape((nx,))

            # control constraints
            ocp.constraints.lbu = -self.u_max
            ocp.constraints.ubu = self.u_max
            ocp.constraints.idxbu = np.array(range(nu))

            # state constraints
            ocp.constraints.lbx = -self.dq_active_max
            ocp.constraints.ubx = self.dq_active_max
            ocp.constraints.idxbx = int(self.nx / 2) + np.array(
                [0, 1, 2 + n_seg[phase_idx]], dtype="int"
            )

            ocp.constraints.lbx_e = -self.dq_active_max
            ocp.constraints.ubx_e = self.dq_active_max
            ocp.constraints.idxbx_e = int(self.nx / 2) + np.array(
                [0, 1, 2 + n_seg[phase_idx]], dtype="int"
            )

            # Enumerating constraints, that should be slacked.
            # Only velocities are constrained in the states
            ocp.constraints.idxsbx = np.array([0, 1, 2])
            ocp.constraints.idxsbx_e = np.array([0, 1, 2])
            ns_angular_velocity = 3

            # safety constraints
            if options.wall_constraint_on:
                ocp.model.con_h_expr = constr_expr[phase_idx]
                ocp.model.con_h_expr_e = constr_expr[phase_idx]
                n_wall_constraints = constr_expr[phase_idx].shape[0]
                self.n_constraints = constr_expr[phase_idx].shape[0]

                n_wall_pos_constraints = n_wall_constraints // 2
                n_wall_speed_constraints = n_wall_constraints // 2

                ns = n_wall_constraints
                nsh = n_wall_constraints  # self.n_constraints
                self.current_slacks = np.zeros((ns,))
                ocp.cost.zl = np.array(
                    [options.w1_slack_speed] * ns_angular_velocity
                    + [options.w1_slack_wall] * n_wall_pos_constraints
                    + [options.w1_slack_speed_wall] * n_wall_speed_constraints
                )
                ocp.cost.Zl = np.array(
                    [options.w2_slack_speed] * ns_angular_velocity
                    + [options.w2_slack_wall] * n_wall_pos_constraints
                    + [options.w2_slack_speed_wall] * n_wall_speed_constraints
                )
                ocp.cost.zu = ocp.cost.zl
                ocp.cost.Zu = ocp.cost.Zl

                ocp.constraints.lh = np.zeros((n_wall_constraints,))
                ocp.constraints.uh = 1e6 * np.ones((n_wall_constraints,))
                ocp.constraints.lh_e = ocp.constraints.lh
                ocp.constraints.uh_e = ocp.constraints.uh

                ocp.constraints.idxsh = np.array(range(n_wall_constraints))
                ocp.constraints.idxsh_e = np.array(range(n_wall_constraints))
            else:
                ocp.cost.zl = np.array([0] * ns_angular_velocity)
                ocp.cost.Zl = np.array([options.w2_slack_speed] * ns_angular_velocity)
                ocp.cost.zu = ocp.cost.zl
                ocp.cost.Zu = ocp.cost.Zl

            ocp.cost.zl_e = ocp.cost.zl
            ocp.cost.zu_e = ocp.cost.zu
            ocp.cost.Zl_e = ocp.cost.Zl
            ocp.cost.Zu_e = ocp.cost.Zu

            # solver options
            ocp.solver_options.qp_solver = (
                "PARTIAL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
            )
            # ocp.solver_options.qp_solver_cond_N = int(
            #    options.n_trans * options.condensing_relative
            # )
            ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
            ocp.solver_options.integrator_type = "IRK"
            ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP_RTI, SQP
            # ocp.solver_options.nlp_solver_max_iter = options.nlp_iter

            # ocp.solver_options.sim_method_num_stages = 2
            # ocp.solver_options.sim_method_num_steps = 2
            # ocp.solver_options.qp_solver_cond_N = n_hor[phase_idx]

            # set parameter values
            p_wall_outside = np.array([0, 1, 0, 0, -1e3, 0])
            ocp.parameter_values = p_wall_outside

            # set prediction horizon
            ocp.solver_options.tf = t_hor[phase_idx]
            #ocp.code_export_directory = self.acados_tmp_dir

            multi_phase_ocp.set_phase(ocp, phase_idx * 2)  # leave space for transition models -> multiply with 2

            if not final_phase:
                # transition
                phase_trans_model = AcadosOcp()
                nx_in = models[phase_idx].x.size()[0]
                nx_out = models[phase_idx + 1].x.size()[0]

                phase_trans_model.model = get_transition_model(n_in=nx_in, n_out=nx_out)
                phase_trans_model.cost.cost_type = 'NONLINEAR_LS'
                phase_trans_model.model.cost_y_expr = phase_trans_model.model.x

                phase_trans_model.cost.W = np.diag(np.zeros((nx_in,)))
                phase_trans_model.cost.yref = np.zeros((nx_in,))

                multi_phase_ocp.set_phase(phase_trans_model, phase_idx * 2 + 1)

            phase_idx += 1

        # Set options
        multi_phase_ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'  # 'FULL_CONDENSING_QPOASES'
        # multi_phase_ocp.solver_options.qp_solver_cond_N = int(
        #        options.n * options.condensing_relative
        #    )
        multi_phase_ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        multi_phase_ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        multi_phase_ocp.solver_options.tf = options.tf
        # multi_phase_ocp.solver_options.nlp_solver_tol_eq = 1e-4
        # multi_phase_ocp.solver_options.nlp_solver_tol_ineq = 1e-4
        # multi_phase_ocp.solver_options.sim_method_num_stages = 2
        # multi_phase_ocp.solver_options.sim_method_num_steps = 2
        multi_phase_ocp.mocp_opts.integrator_type = ['IRK', 'DISCRETE', 'IRK']
       # multi_phase_ocp.code_export_directory = self.acados_tmp_dir

        self.acados_ocp_solver = (
            AcadosOcpSolver(multi_phase_ocp,
                            json_file="acados_ocp_mpc_phases.json"))

    def reset(self):
        self.debug_timings = []
        self.iteration_counter = 0

    def set_wall_parameters(self, w: np.ndarray, b: np.ndarray):
        """
        Set wall parameters such that w.T @ (x_ee @ b) >= 0
        @param w: vector in the direction of feasibility
        @param b: distance to a point on the wall
        """
        p = np.hstack((w, b))
        for ii in range(self.options.n_trans):
            self.acados_ocp_solver.set(ii, "p", p)

        for ii in range(self.options.n_trans + 1, self.options.n):
            self.acados_ocp_solver.set(ii, "p", p)

    def set_reference_point(self, q: np.ndarray, p_ee_ref: np.ndarray):
        """
        Sets a reference point which the method "compute_torque" will then track and stabilize.

        @param q: Current estimated/measured joint positions
        @param p_ee_ref: Endefector Cartesian Endefector reference position
        """

        self.p_ee_ref = p_ee_ref
        x_ref, u_ref = compute_reference_state_and_input(self.fa_model_p1, q, p_ee_ref)

        if len(p_ee_ref.shape) < 2:
            p_ee_ref = np.expand_dims(p_ee_ref, 1)
        if len(x_ref.shape) < 2:
            x_ref = np.expand_dims(x_ref, 1)
        if len(u_ref.shape) < 2:
            u_ref = np.expand_dims(u_ref, 1)

        yref = np.vstack((x_ref, u_ref, p_ee_ref)).flatten()
        yref_e = np.vstack((x_ref, p_ee_ref)).flatten()

        for stage in range(self.options.n_trans):
            self.acados_ocp_solver.cost_set(stage, "yref", yref)

        # transition two
        q_p2 = translate_config(q, self.options.n_seg_p1, self.options.n_seg_p2)
        x_ref, u_ref = compute_reference_state_and_input(self.fa_model_p2, q_p2, p_ee_ref)

        if len(p_ee_ref.shape) < 2:
            p_ee_ref = np.expand_dims(p_ee_ref, 1)
        if len(x_ref.shape) < 2:
            x_ref = np.expand_dims(x_ref, 1)
        if len(u_ref.shape) < 2:
            u_ref = np.expand_dims(u_ref, 1)

        yref = np.vstack((x_ref, u_ref, p_ee_ref)).flatten()
        yref_e = np.vstack((x_ref, p_ee_ref)).flatten()

        for stage in range(self.options.n_trans + 1, self.options.n + 1):
            self.acados_ocp_solver.cost_set(stage, "yref", yref)

        self.acados_ocp_solver.cost_set(self.options.n + 1, "yref", yref_e)

    def compute_torques(self, q: np.ndarray, dq: np.ndarray, t: float = None, y=None):
        """
        Main control loop function that computes the torques at a specific time. The time is only required if a
        reference trajectory is used.
        @param q: Estimated/Measured position states
        @param dq: Estimated/measured velocity states
        @param t: current time (related to reference specification)
        @return: torques tau
        """
        # set initial state
        xcurrent = np.vstack((q, dq))
        self.acados_ocp_solver.set(0, "lbx", xcurrent)
        self.acados_ocp_solver.set(0, "ubx", xcurrent)

        # If we specified a reference trajectory, we need to compute the current reference points for mpc
        if t is not None and self.inter_t2q is not None and self.inter_t2dq is not None:
            t_vec = np.linspace(t, t + self.options.tf, self.options.n + 1)
            q_ref_vec = self.inter_t2q(t_vec)
            dq_ref_vec = self.inter_t2dq(t_vec)
            pee_ref_vec = self.inter_pee(t_vec)
            x_vec = np.hstack((q_ref_vec, dq_ref_vec))
            u_ref = self.inter_t2u(t_vec)

            for stage in range(self.options.n):
                yref = np.vstack(
                    (
                        np.expand_dims(x_vec[stage, :], 1),
                        np.expand_dims(u_ref[stage, :], 1),
                        np.expand_dims(pee_ref_vec[stage, :], 1),
                    )
                ).flatten()
                self.acados_ocp_solver.cost_set(stage, "yref", yref)
            stage = self.options.n
            yref_e = np.vstack(
                (
                    np.expand_dims(x_vec[stage, :], 1),
                    np.expand_dims(pee_ref_vec[stage, :], 1),
                )
            ).flatten()
            self.acados_ocp_solver.cost_set(self.options.n, "yref", yref_e)

        # acados solve NLP
        status = self.acados_ocp_solver.solve()

        # Get timing result
        self.debug_timings.append(self.acados_ocp_solver.get_stats("time_tot"))

        # Check for errors in acados
        if status != 0:
            print(
                "acados returned status {} in time step {}".format(
                    status, self.iteration_counter
                )
            )
        self.iteration_counter += 1

        # Retrieve control u
        if status == 0:
            u_output = self.acados_ocp_solver.get(0, "u")
            self.last_U = np.zeros((self.nu, self.options.n - 1))
            for i in range(self.options.n - 1):
                u_out = self.acados_ocp_solver.get(i, "u")
                if u_out.shape[0] == 3:
                    self.last_U[:, i] = u_out
        else:
            self.acados_ocp_solver.reset()
            if self.last_U is not None:
                u_output = self.last_U[:, 1]
                self.last_U[:, :-1] = self.last_U[:, 1:]
            else:
                u_output = np.zeros((self.nu,))

        return u_output

    def get_timing_statistics(self) -> Tuple[float, float, float, float]:
        """
        Get timing statistics of all mpc evaluations
        """
        timing_array = np.array(self.debug_timings)
        t_mean = float(np.mean(timing_array))
        t_std = float(np.std(timing_array))
        t_max = float(np.max(timing_array))
        t_min = float(np.min(timing_array))
        return t_mean, t_std, t_min, t_max

    def get_last_computation_time(self):
        """
        Return most recent computation time. Empty, if not started.
        """
        return self.debug_timings[-1]
