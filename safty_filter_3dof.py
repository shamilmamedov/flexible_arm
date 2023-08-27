from copy import copy
from dataclasses import dataclass
import time
from tempfile import mkdtemp

import numpy as np
import scipy
from scipy.interpolate import interp1d
from controller import BaseController, OfflineController
from typing import TYPE_CHECKING, Tuple
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver

from estimator import ExtendedKalmanFilter
from poly5_planner import initial_guess_for_active_joints, get_reference_for_all_joints
from simulation import Simulator
from envs.flexible_arm_3dof import SymbolicFlexibleArm3DOF, get_rest_configuration

# Avoid circular imports with type checking
if TYPE_CHECKING:
    from envs.flexible_arm_3dof import FlexibleArm3DOF


@dataclass
class SafetyFilter3dofOptions:
    """
    Dataclass for MPC options
    """

    def __init__(self):
        self.n_seg: int = 1  # n_links corresponds to (n+1)*2 states
        self.n: int = 20  # number of discretization points
        self.tf: float = 0.2  # time horizon
        self.nlp_iter: int = 100  # number of iterations of the nonlinear solver
        self.z_diag: np.ndarray = np.array([0] * 3) * 1e1
        self.z_e_diag: np.ndarray = np.array([0] * 3) * 1e3
        self.r_diag: np.ndarray = np.array([1.0, 1.0, 1.0]) * 1
        self.r_diag_rollout: np.ndarray = np.array([1.0, 1.0, 1.0]) * 1e-5
        self.w2_slack_speed: float = 1e3
        self.w2_slack_wall: float = 1e5
        self.w1_slack_wall: float = 1
        self.w_reg_dq: float = 1e-2
        self.w_reg_dq_terminal: float = 1e-2
        self.wall_constraint_on: bool = True  # choose whether we activate the wall constraint
        # todo: check how to tune weights

    def get_sampling_time(self) -> float:
        return self.tf / self.n


class SafetyFilter3Dof:
    """
    Safety filter
    """

    def __init__(
            self,
            x0: np.ndarray = None,
            x0_ee: np.ndarray = None,
            options: SafetyFilter3dofOptions = SafetyFilter3dofOptions(),
    ):
        if x0 is None:
            x0 = np.zeros((2 * (1 + 2 * (1 + options.n_seg)), 1))
        if x0_ee is None:
            x0_ee = np.zeros((3, 1))

        fa_model = SymbolicFlexibleArm3DOF(n_seg=options.n_seg)
        self.u_max = fa_model.tau_max  # [Nm]
        self.dq_active_max = fa_model.dqa_max
        self.fa_model = fa_model
        model, constraint_expr = fa_model.get_acados_model_safety()
        self.model = model
        self.options = options
        self.debug_timings = []
        self.debug_total_timings = []
        self.iteration_counter = 0
        self.inter_t2q = None
        self.inter_t2dq = None
        self.inter_pee = None
        self.u_pre_safe = None

        # set up simulator for initial state
        integrator = "RK45"
        self.offline_controller = OfflineController()

        # create ocp object to formulate the OCP
        ocp = AcadosOcp()
        # set model
        ocp.model = model

        # OCP parameter adjustment
        nx = model.x.size()[0]
        nu = model.u.size()[0]
        nz = model.z.size()[0]
        ny = nx + nu + nz
        ny_e = nx + nz
        self.nu = nu
        self.nx = nx
        self.nz = nz

        # define constraints
        # ocp.model.con_h_expr = constraint_expr

        # some checks
        assert nu == options.r_diag.shape[0]
        assert nz == options.z_diag.shape[0] == options.z_e_diag.shape[0]

        # set dimensions
        ocp.dims.N = options.n

        # set cost module
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"

        # setting state weights
        q_diag = np.zeros((2 + 4 * (options.n_seg + 1),))
        q_e_diag = np.zeros((2 + 4 * (options.n_seg + 1),))
        q_diag[0: int(len(q_diag) / 2)] = 0
        q_diag[int(len(q_diag) / 2):] = options.w_reg_dq
        q_e_diag[0: int(len(q_e_diag) / 2)] = 0
        q_e_diag[int(len(q_e_diag) / 2):] = options.w_reg_dq_terminal
        Q = np.diagflat(q_diag)
        Q_e = np.diagflat(q_e_diag)

        R = np.diagflat(options.r_diag)
        R_rollout = np.diagflat(options.r_diag_rollout)
        Z = np.diagflat(options.z_diag)
        Z_e = np.diagflat(options.z_e_diag)

        # state constraints
        ocp.constraints.lbx = -self.dq_active_max
        ocp.constraints.ubx = self.dq_active_max
        ocp.constraints.idxbx = int(self.nx / 2) + np.array([0, 1, 2 + options.n_seg], dtype="int")

        ocp.constraints.lbx_e = -self.dq_active_max
        ocp.constraints.ubx_e = self.dq_active_max
        ocp.constraints.idxbx_e = int(self.nx / 2) + np.array([0, 1, 2 + options.n_seg], dtype="int")
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
            ocp.cost.zl = np.array([0] * 3 +
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
        ocp.cost.Vx_e[:nx, :nx] = np.ones(nx)

        # Vz_e = np.zeros((ny_e, nz))
        # Vz_e[nx:, :] = np.eye(nz)
        # ocp.cost.Vz_e = Vz_e

        ocp.model.name = "safety_" + str(options.n) + "_" + str(options.n_seg)
        ocp.code_export_directory = mkdtemp()

        x_goal = x0
        x_goal_cartesian = x0_ee  # np.expand_dims(np.array([x_cartesian, y_cartesian, z_cartesian]), 1)
        ocp.cost.yref = np.vstack((x_goal, np.zeros((nu, 1)), x_goal_cartesian)).flatten()
        ocp.cost.yref_e = np.vstack((x_goal, x_goal_cartesian)).flatten()
        self.x0_ee = x0_ee

        # set constraints
        umax = self.u_max * np.ones((nu,))
        ocp.constraints.constr_type = "BGH"
        ocp.constraints.lbu = -umax
        ocp.constraints.ubu = umax
        ocp.constraints.x0 = x0.reshape((nx,))
        ocp.constraints.idxbu = np.array(range(nu))
        # ocp.constraints.idxbx = np.array([0])
        # ocp.constraints.lbx = -np.array([np.pi / 2])
        # ocp.constraints.ubx = np.array([np.pi / 2])

        # solver options
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "IRK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP_RTI, SQP
        ocp.solver_options.nlp_solver_max_iter = options.nlp_iter

        ocp.solver_options.sim_method_num_stages = 2
        ocp.solver_options.sim_method_num_steps = 2
        ocp.solver_options.qp_solver_cond_N = int(1 * options.n)

        # set parameter values
        p_wall_outside = np.array([0, 1, 0, 0, -1e3, 0])
        ocp.parameter_values = p_wall_outside

        # set prediction horizon
        ocp.solver_options.tf = options.tf

        ocp.code_export_directory = "c_generated_code_safety_filter"

        self.acados_ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp_sf.json")

        # set costs only for the first stage, ignore the rest
        for stage in range(self.options.n):
            if stage == 0:
                self.acados_ocp_solver.cost_set(stage, "W", scipy.linalg.block_diag(Q, R, np.zeros_like(Z)))
            else:
                self.acados_ocp_solver.cost_set(stage, "W", scipy.linalg.block_diag(Q, R_rollout, np.zeros_like(Z)))
        self.acados_ocp_solver.cost_set(self.options.n, "W", scipy.linalg.block_diag(Q_e, np.zeros_like(Z)))

        # Create model instances
        est_model = SymbolicFlexibleArm3DOF(options.n_seg, dt=self.options.tf / options.n, integrator="cvodes")

        # Design estimator
        # initial covariance matrix
        p0_q = [0.05] * est_model.nq  # 0.05, 0.1
        p0_dq = [1e-3] * est_model.nq
        P0 = np.diag([*p0_q, *p0_dq])
        # process noise covariance
        q_q = [1e-4, *[1e-3] * (est_model.nq - 1)]
        q_dq = [1e-1, *[5e-1] * (est_model.nq - 1)]  # 5e-1, 10e-1
        Q = np.diag([*q_q, *q_dq])
        # measurement noise covaiance
        r_q, r_dq, r_pee = [3e-5] * 3, [5e-2] * 3, [1e-3] * 3
        R = 100 * np.diag([*r_q, *r_dq, *r_pee])

        # initial state for the estimator
        qa0_est = np.array([0, 0, 0])
        q0_est = get_rest_configuration(qa0_est, options.n_seg)
        dq0_est = np.zeros_like(q0_est)
        x0_est = np.vstack((q0_est, dq0_est))

        self.E = ExtendedKalmanFilter(est_model, x0_est, P0, Q, R)

    def set_wall_parameters(self, w: np.ndarray, b: np.ndarray):
        """
        Set wall parameters such that w.T @ (x_ee @ b) >= 0
        @param w: vector in the direction of feasibility
        @param b: distance to a point on the wall
        """
        p = np.hstack((w, b))
        for ii in range(self.options.n):
            self.acados_ocp_solver.set(ii, "p", p)

    def set_reference_point(self, x_ref: np.ndarray, p_ee_ref: np.ndarray, u_ref: np.array):
        """
        Sets a reference point which the mehtod "compute_torque" will then track and stabilize.

        @param x_ref: States q and dq of the reference position
        @param p_ee_ref: Endefector reference position
        @param u_ref: Torques at endeffector position. Can be set to zero.
        """
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

    def reset(self):
        self.debug_timings = []
        self.iteration_counter = 0

    def filter(self, u0: np.ndarray, y: np.ndarray):
        # first estimate state
        y = np.expand_dims(y, 1)
        if self.u_pre_safe is None:
            qa = y[:3]
            x_hat = np.zeros_like(self.E.x_hat)
            idx_qa = [0, 1, 2 + self.options.n_seg]
            x_hat[idx_qa] = qa
            self.E.x_hat = x_hat
            self.x_hat = self.E.estimate(y).flatten()
        else:
            self.x_hat = self.E.estimate(y, self.u_pre_safe).flatten()

        # set initial state
        start_time = time.time()
        xcurrent = self.x_hat  # np.vstack((q, dq))
        self.acados_ocp_solver.set(0, "lbx", xcurrent)
        self.acados_ocp_solver.set(0, "ubx", xcurrent)

        # simulate states
        # set u0 reference
        stage = 0
        if u0.shape.__len__() <= 1:
            u0 = np.expand_dims(u0.transpose(), 1)
        else:
            u0 = u0.transpose()
        yref = np.vstack((np.zeros((self.nx, 1)), u0, np.zeros((self.nz, 1)))).flatten()
        self.acados_ocp_solver.cost_set(stage, "yref", yref)

        # acados solve NLP
        status = self.acados_ocp_solver.solve()

        # Get timing result
        self.debug_timings.append(self.acados_ocp_solver.get_stats("time_tot")[0])

        # Check for errors in acados
        if status != 0:
            raise Exception(
                "acados returned status {} in time step {}. Exiting.".format(status, self.iteration_counter)
            )
        self.iteration_counter += 1

        # Retrieve control u
        u_output = self.acados_ocp_solver.get(0, "u")

        self.debug_total_timings.append(time.time() - start_time)
        self.u_pre_safe = u_output
        return u_output

    def get_timing_statistics(self, mode=0) -> Tuple[float, float, float, float]:
        if mode == 0:
            timing_array = np.array(self.debug_timings)
        else:
            timing_array = np.array(self.debug_total_timings)
        t_mean = float(np.mean(timing_array))
        t_std = float(np.std(timing_array))
        t_max = float(np.max(timing_array))
        t_min = float(np.min(timing_array))
        return t_mean, t_std, t_min, t_max


def get_safe_controller_class(base_controller_class, safety_filter: SafetyFilter3Dof):
    """
    This fancy function changes the "compute torque" function of a controller in order to include a safety filter
    @param base_controller_class: base controller unsafe class
    @param safety_filter: object of a safety filter
    @return: base controller safe class
    """

    class ControllerSafetyWrapper(base_controller_class):
        def __init__(self, *args, **kwargs):
            super(ControllerSafetyWrapper, self).__init__(*args, **kwargs)
            self.u_pre_safe = None

        def set_reference_point(self, x_ref: np.ndarray, p_ee_ref: np.ndarray, u_ref: np.array):
            safety_filter.set_reference_point(x_ref, p_ee_ref, u_ref)

        def compute_torques(self, q, dq, t=None, y=None):
            assert y is not None
            u = super(ControllerSafetyWrapper, self).compute_torques(q, dq, t=t)
            u_safe = safety_filter.filter(u0=u, y=y)
            # print(u - u_safe)
            self.u_pre_safe = u_safe
            return u_safe

        def get_timing_statistics(self):
            return safety_filter.get_timing_statistics()

    return ControllerSafetyWrapper
