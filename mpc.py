from copy import copy
from dataclasses import dataclass

import numpy as np
import scipy

from controller import BaseController
from typing import TYPE_CHECKING, Tuple
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver

# Avoid circular imports with type checking
if TYPE_CHECKING:
    from flexible_arm import SymbolicFlexibleArm


@dataclass
class MpcOptions:
    n: int = 30  # number of discretization points
    tf: float = 2  # time horizon
    nlp_iter: int = 100  # number of iterations of the nonlinear solver

    def get_sampling_time(self) -> float:
        return self.tf / self.n


class Mpc(BaseController):
    def __init__(self, model: "SymbolicFlexibleArm",
                 x0: np.ndarray,
                 options: MpcOptions = MpcOptions()):
        self.u_max = model.maximum_input_torque
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

        # set dimensions
        ocp.dims.N = options.n

        # set cost module
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'

        q_diag = np.zeros((nx, 1))
        q_diag[0:int(nx / 2)] = 1

        Q = 0 * np.diagflat(q_diag)
        Q_e = 0 * Q
        R = np.diagflat(1e-2 * np.ones((nu, 1)))
        Z = 1e2 * np.diagflat(np.ones((nz, 1)))
        Z_e = 10 * Z

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

        # todo: reference position is hardcoded here for now
        x_goal = np.zeros((nx, 1))
        x_goal[0] = np.pi / 4
        phi =  np.pi / 2 + 0.3
        x_cartesian = np.cos(phi) * 0.4
        y_cartesian = np.sin(phi) * 0.4
        x_goal_cartesian = np.expand_dims(np.array([x_cartesian, y_cartesian]), 1)
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
