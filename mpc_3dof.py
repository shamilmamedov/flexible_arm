from dataclasses import dataclass
from tempfile import mkdtemp

import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline
from controller import BaseController
from typing import TYPE_CHECKING, Tuple
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from poly5_planner import initial_guess_for_active_joints, get_reference_for_all_joints
from envs.flexible_arm_3dof import get_rest_configuration, inverse_kinematics_rb

# Avoid circular imports with type checking
if TYPE_CHECKING:
    from envs.flexible_arm_3dof import FlexibleArm3DOF, SymbolicFlexibleArm3DOF

Q_QA = 0.01  # penalty on active joints positions # 0.1, 1
Q_QP = 0.01  # penalty on passive joints positions # 0.1, 0.001
Q_DQA = 0.1  # penalty on active joints velocities # 10., 1., 0.1,
Q_DQP = 10  # penalty on passive joints velocities # 0.001, 0.1
Q_DQA_E = 1.0  # penalty on terminal active joints velocities
Q_QA_E = 0.1  # penalty on terminal active joints velocities


@dataclass
class Mpc3dofOptions:
    """
    Dataclass for MPC options
    """

    def __init__(self, n_seg: int, tf: float = 2, n: int = 30):
        self.n_seg: int = n_seg  # n_seg corresponds to (1 + 2 * (n_seg + 1))*2 states
        self.n: int = n  # number of discretization points
        self.tf: float = tf  # time horizon
        self.nlp_iter: int = 50  # number of iterations of the nonlinear solver
        self.condensing_relative: float = 1  # relative factor of condensing [0-1]
        self.wall_constraint_on: bool = (
            True  # choose whether we activate the wall constraint
        )

        # States are ordered for each link
        self.q_diag: np.ndarray = np.array(
            [Q_QA] * (2)
            + [Q_QP] * (self.n_seg)  # qa1 and qa2
            + [Q_QA] * (1)  # qp 1st link
            + [Q_QP] * (self.n_seg)  # qa3
            + [Q_DQA] * (2)  # qp 2nd link
            + [Q_DQP] * (self.n_seg)  # dqa1 and dqa2
            + [Q_DQA] * (1)  # dqp 1st link
            + [Q_DQP] * (self.n_seg)  # dqa3
        )  # dqp 2nd link
        self.q_e_diag: np.ndarray = np.array(
            [Q_QA_E] * (2)
            + [Q_QP] * (self.n_seg)  # qa1 and qa2
            + [Q_QA_E] * (1)  # qp 1st link
            + [Q_QP] * (self.n_seg)  # qa3
            + [Q_DQA_E] * (2)  # qp 2nd link
            + [Q_DQP] * (self.n_seg)  # dqa1 and dqa2
            + [Q_DQA_E] * (1)  # dqp 1st link
            + [Q_DQP] * (self.n_seg)  # dqa3
        )  # dqp 2nd link
        self.z_diag: np.ndarray = np.array([1] * 3) * 3e3
        self.z_e_diag: np.ndarray = np.array([1] * 3) * 1e4
        self.r_diag: np.ndarray = np.array([1e0, 10e0, 10e0]) * 1e-1
        self.w2_slack_speed: float = 1e6
        self.w2_slack_wall: float = 1e5
        self.w1_slack_wall: float = 1e1

    def get_sampling_time(self) -> float:
        return self.tf / self.n


class Mpc3Dof(BaseController):
    """
    Controller class for 3 dof flexible link model based on acados
    """

    def __init__(
            self,
            model: "SymbolicFlexibleArm3DOF",
            x0: np.ndarray = None,
            pee_0: np.ndarray = None,
            options: Mpc3dofOptions = Mpc3dofOptions(n_seg=1),
    ):
        """
        :parameter x0: initial state vector
        :parameter pee_0: initial end-effector position
        :parameter options: a class with options
        """
        if x0 is None:
            x0 = np.zeros((2 * (1 + 2 * (1 + options.n_seg)), 1))
        if pee_0 is None:
            pee_0 = np.zeros((3, 1))
        self.u_max = model.tau_max  # [Nm]
        self.dq_active_max = model.dqa_max  # [rad/s]

        self.fa_model = model
        model, constraint_expr = model.get_acados_model_safety()
        self.model = model
        self.options = options
        self.iteration_counter = 0
        self.inter_t2q = None
        self.inter_t2dq = None
        self.inter_pee = None

        # create ocp object to formulate the OCP
        ocp = AcadosOcp()
        ocp.model = model  # set model

        # OCP parameter adjustment
        nx = model.x.size()[0]
        nu = model.u.size()[0]
        nz = model.z.size()[0]
        n_p = model.p.size()[0]
        ny = nx + nu + nz
        ny_e = nx + nz
        self.nu = nu
        self.nx = nx

        # some checks
        assert nx == options.q_diag.shape[0] == options.q_e_diag.shape[0]
        assert nu == options.r_diag.shape[0]
        assert nz == options.z_diag.shape[0] == options.z_e_diag.shape[0]

        ocp.model.name = "mpc_" + str(options.n) + "_" + str(options.n_seg)
        ocp.code_export_directory = mkdtemp()

        # set dimensions
        ocp.dims.N = options.n

        # set cost module
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"

        Q = np.diagflat(options.q_diag)
        Q_e = np.diagflat(options.q_e_diag)
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

        x_goal = x0
        x_goal_cartesian = pee_0  # np.expand_dims(np.array([x_cartesian, y_cartesian, z_cartesian]), 1)
        ocp.cost.yref = np.vstack(
            (x_goal, np.zeros((nu, 1)), x_goal_cartesian)
        ).flatten()
        ocp.cost.yref_e = np.vstack((x_goal, x_goal_cartesian)).flatten()

        # general constraints
        ocp.constraints.constr_type = "BGH"
        ocp.constraints.x0 = x0.reshape((nx,))

        # control constraints
        ocp.constraints.lbu = -self.u_max
        ocp.constraints.ubu = self.u_max
        ocp.constraints.idxbu = np.array(range(nu))

        # state constraints
        ocp.constraints.lbx = -self.dq_active_max
        ocp.constraints.ubx = self.dq_active_max
        ocp.constraints.idxbx = int(self.nx / 2) + np.array(
            [0, 1, 2 + options.n_seg], dtype="int"
        )

        ocp.constraints.lbx_e = -self.dq_active_max
        ocp.constraints.ubx_e = self.dq_active_max
        ocp.constraints.idxbx_e = int(self.nx / 2) + np.array(
            [0, 1, 2 + options.n_seg], dtype="int"
        )

        # Enumerating constraints, that should be slacked.
        # Only velocities are constrained in the states
        ocp.constraints.idxsbx = np.array([0, 1, 2])
        ocp.constraints.idxsbx_e = np.array([0, 1, 2])

        # safety constraints
        if options.wall_constraint_on:
            ocp.model.con_h_expr = constraint_expr
            ocp.model.con_h_expr_e = constraint_expr
            n_wall_constraints = constraint_expr.shape[0]
            self.n_constraints = constraint_expr.shape[0]
            ns = n_wall_constraints
            nsh = n_wall_constraints  # self.n_constraints
            self.current_slacks = np.zeros((ns,))
            ocp.cost.zl = np.array([10 ** 3] * 3 +
                                   [options.w1_slack_wall] * n_wall_constraints)
            ocp.cost.Zl = np.array(
                [options.w2_slack_speed] * 3
                + [options.w2_slack_wall] * n_wall_constraints
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
            ocp.cost.zl = np.array([0] * 3)
            ocp.cost.Zl = np.array([options.w2_slack_speed] * 3)
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
        ocp.solver_options.qp_solver_cond_N = int(
            options.n * options.condensing_relative
        )
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "IRK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP_RTI, SQP
        ocp.solver_options.nlp_solver_max_iter = options.nlp_iter

        ocp.solver_options.sim_method_num_stages = 2
        ocp.solver_options.sim_method_num_steps = 2
        ocp.solver_options.qp_solver_cond_N = options.n

        # set parameter values
        p_wall_outside = np.array([0, 1, 0, 0, -1e3, 0])
        ocp.parameter_values = p_wall_outside

        # set prediction horizon
        ocp.solver_options.tf = options.tf

        ocp.code_export_directory = "c_generated_code_mpc"
        self.acados_ocp_solver = AcadosOcpSolver(
            ocp, json_file="acados_ocp_mpc.json"
        )

    def reset(self):
        self.debug_timings = []
        self.iteration_counter = 0

    def _get_x_ref(self, p_ee_ref: np.ndarray):
        """
        Compute reference states genaralized coordinates (angles q, angular velocities dq)
        related to MPC model out of endeffetor states.
        @param p_ee_ref: Endefector Cartesian reference position
        """

        qa_t0 = np.array([np.pi / 2, np.pi / 10, -np.pi / 8])
        q_t0_guess = get_rest_configuration(qa_t0, self.options.n_seg)

        qa_guess = q_t0_guess[self.fa_model.qa_idx]

        qa_guess = inverse_kinematics_rb(p_ee_ref, q_guess=qa_guess)
        qa_0 = qa_guess.flatten()
        q_0 = get_rest_configuration(qa_0, self.fa_model.n_seg).T

        dq_0 = np.zeros_like(q_0)
        x_0 = np.hstack((q_0, dq_0))
        return x_0.T

    def set_wall_parameters(self, w: np.ndarray, b: np.ndarray):
        """
        Set wall parameters such that w.T @ (x_ee @ b) >= 0
        @param w: vector in the direction of feasibility
        @param b: distance to a point on the wall
        """
        p = np.hstack((w, b))
        for ii in range(self.options.n):
            self.acados_ocp_solver.set(ii, "p", p)

    def set_reference_point(self, p_ee_ref: np.ndarray):
        """
        Sets a reference point which the mehtod "compute_torque" will then track and stabilize.

        @param x_ref: States q and dq of the reference position
        @param p_ee_ref: Endefector Cartesian Endefector reference position
        @param u_ref: Torques at endeffector position. Can be set to zero.
        """
        # get reference states in generalized coordinates from ee position
        x_ref = self._get_x_ref(p_ee_ref)

        # set reference torques to zero
        u_ref = np.array([[0.0], [0.0], [0.0]])

        if len(p_ee_ref.shape) < 2:
            p_ee_ref = np.expand_dims(p_ee_ref, 1)
        if len(x_ref.shape) < 2:
            x_ref = np.expand_dims(x_ref, 1)
        if len(u_ref.shape) < 2:
            u_ref = np.expand_dims(u_ref, 1)

        yref = np.vstack((x_ref, u_ref, p_ee_ref)).flatten()
        yref_e = np.vstack((x_ref, p_ee_ref)).flatten()

        for stage in range(self.options.n):
            self.acados_ocp_solver.cost_set(stage, "yref", yref)
        self.acados_ocp_solver.cost_set(self.options.n, "yref", yref_e)

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
        self.debug_timings.append(self.acados_ocp_solver.get_stats("time_tot")[0])

        # Check for errors in acados
        if status != 0 and status != 2:
            raise Exception(
                "acados returned status {} in time step {}. Exiting.".format(
                    status, self.iteration_counter
                )
            )
        self.iteration_counter += 1

        # Retrieve control u
        u_output = self.acados_ocp_solver.get(0, "u")
        return u_output
