from copy import copy
from dataclasses import dataclass

import numpy as np
import scipy

from controller import BaseController
from typing import TYPE_CHECKING, Tuple
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver

# Avoid circular imports with type checking
if TYPE_CHECKING:
    from flexible_arm_3dof import FlexibleArm3DOF, SymbolicFlexibleArm3DOF


@dataclass
class Mpc3dofOptions:

    def __init__(self, n_links: int):
        self.n_links: int = n_links  # n_links corresponds to (n+1)*2 states
        self.n: int = 100  # number of discretization points
        self.tf: float = 3  # time horizon
        self.nlp_iter: int = 100  # number of iterations of the nonlinear solver

        # States are ordered for each link
        self.q_diag: np.ndarray = np.array([1] * (1) + [0] * (1) + \
                                           [1] * (self.n_links + 1) + [0] * (self.n_links + 1) + \
                                           [1] * (self.n_links + 1) + [0] * (self.n_links + 1))
        self.q_e_diag: np.ndarray = np.array([1] * (1) + [0] * (1) + \
                                             [1] * (self.n_links + 1) + [0] * (self.n_links + 1) + \
                                             [1] * (self.n_links + 1) + [0] * (self.n_links + 1))
        self.z_diag: np.ndarray = np.array([1] * 3) * 1e1
        self.z_e_diag: np.ndarray = np.array([1] * 3) * 1e3
        self.r_diag: np.ndarray = np.array([1e-2, 1e-2, 1e-2])

    def get_sampling_time(self) -> float:
        return self.tf / self.n


class Mpc3Dof(BaseController):
    def __init__(self, model: "SymbolicFlexibleArm3DOF",
                 x0: np.ndarray,
                 x0_ee: np.ndarray,
                 options: Mpc3dofOptions = Mpc3dofOptions(n_links=3)):
        self.u_max = 1e6
        self.fa_model = model
        model = model.get_acados_model()
        self.model = model
        self.options = options
        self.debug_timings = []
        self.iteration_counter = 0

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

        # some checks
        assert (nx == options.q_diag.shape[0] == options.q_e_diag.shape[0])
        assert (nu == options.r_diag.shape[0])
        assert (nz == options.z_diag.shape[0] == options.z_e_diag.shape[0])

        # set dimensions
        ocp.dims.N = options.n

        # set cost module
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'

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
        Vu[nx:nx + nu, :] = np.eye(nu)
        ocp.cost.Vu = Vu

        Vz = np.zeros((ny, nz))
        Vz[nx + nu:, :] = np.eye(nz)
        ocp.cost.Vz = Vz

        ocp.cost.Vx_e = np.zeros((ny_e, nx))
        ocp.cost.Vx_e[:nx, :nx] = np.ones(nx)

        # Vz_e = np.zeros((ny_e, nz))
        # Vz_e[nx:, :] = np.eye(nz)
        # ocp.cost.Vz_e = Vz_e

        x_goal = x0
        x_goal_cartesian = x0_ee  # np.expand_dims(np.array([x_cartesian, y_cartesian, z_cartesian]), 1)
        ocp.cost.yref = np.vstack((x_goal, np.zeros((nu, 1)), x_goal_cartesian)).flatten()
        ocp.cost.yref_e = np.vstack((x_goal, x_goal_cartesian)).flatten()

        # set constraints
        umax = self.u_max * np.ones((nu,))

        ocp.constraints.constr_type = 'BGH'
        ocp.constraints.lbu = -umax
        ocp.constraints.ubu = umax
        ocp.constraints.x0 = x0.reshape((nx,))
        ocp.constraints.idxbu = np.array(range(nu))

        # solver options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'  # FULL_CONDENSING_QPOASES
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'IRK'
        ocp.solver_options.nlp_solver_type = 'SQP'  # SQP_RTI
        ocp.solver_options.nlp_solver_max_iter = options.nlp_iter

        ocp.solver_options.sim_method_num_stages = 2
        ocp.solver_options.sim_method_num_steps = 2
        ocp.solver_options.qp_solver_cond_N = options.n

        # set prediction horizon
        ocp.solver_options.tf = options.tf

        self.acados_ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp_' + model.name + '.json')

    def reset(self):
        self.debug_timings = []
        self.iteration_counter = 0

    def set_reference_cartesian(self, x_ref: np.ndarray, p_ee_ref: np.ndarray, u_ref: np.array):
        if len(p_ee_ref.shape) < 1:
            p_ee_ref = np.expand_dims(p_ee_ref, 1)
        if len(x_ref.shape) < 1:
            x_ref = np.expand_dims(x_ref, 1)
        if len(u_ref.shape) < 1:
            u_ref = np.expand_dims(u_ref, 1)

        yref = np.vstack((x_ref, u_ref, p_ee_ref)).flatten()
        yref_e = np.vstack((x_ref, p_ee_ref)).flatten()

        for stage in range(self.options.n):
            self.acados_ocp_solver.cost_set(stage, "yref", yref)
        self.acados_ocp_solver.cost_set(self.options.n, "yref", yref_e)

    def compute_torques(self, q: np.ndarray, dq: np.ndarray):
        xcurrent = np.vstack((q, dq))
        # solve ocp
        self.acados_ocp_solver.set(0, "lbx", xcurrent)
        self.acados_ocp_solver.set(0, "ubx", xcurrent)

        status = self.acados_ocp_solver.solve()
        self.debug_timings.append(self.acados_ocp_solver.get_stats("time_tot")[0])

        if status != 0:
            raise Exception('acados returned status {} in time step {}. Exiting.'.format(status,
                                                                                         self.iteration_counter))
        self.iteration_counter += 1
        u_output = self.acados_ocp_solver.get(0, "u")
        return u_output

    def get_timing_statistics(self) -> Tuple[float, float, float, float]:
        timing_array = np.array(self.debug_timings)
        t_mean = float(np.mean(timing_array))
        t_std = float(np.std(timing_array))
        t_max = float(np.max(timing_array))
        t_min = float(np.min(timing_array))
        return t_mean, t_std, t_min, t_max
