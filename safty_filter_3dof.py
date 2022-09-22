from dataclasses import dataclass
import time

import numpy as np
import scipy
from scipy.interpolate import interp1d
from controller import BaseController, OfflineController
from typing import TYPE_CHECKING, Tuple
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from poly5_planner import initial_guess_for_active_joints, get_reference_for_all_joints
from simulation import Simulator

# Avoid circular imports with type checking
if TYPE_CHECKING:
    from flexible_arm_3dof import FlexibleArm3DOF, SymbolicFlexibleArm3DOF


@dataclass
class SafetyFilter3dofOptions:
    """
    Dataclass for MPC options
    """

    def __init__(self, n_links: int):
        self.n_links: int = n_links  # n_links corresponds to (n+1)*2 states
        self.n: int = 10  # number of discretization points
        self.tf: float = 0.1  # time horizon
        self.nlp_iter: int = 100  # number of iterations of the nonlinear solver

        # States are ordered for each link
        self.q_diag: np.ndarray = np.array([0] * (1) + [0] * (1) + \
                                           [0] * (self.n_links + 1) + [0] * (self.n_links + 1) + \
                                           [0] * (self.n_links + 1) + [0] * (self.n_links + 1))
        self.q_e_diag: np.ndarray = np.array([0] * (1) + [1] * (1) + \
                                             [0] * (self.n_links + 1) + [0] * (self.n_links + 1) + \
                                             [0] * (self.n_links + 1) + [0] * (self.n_links + 1))
        self.z_diag: np.ndarray = np.array([0] * 3) * 1e1
        self.z_e_diag: np.ndarray = np.array([0] * 3) * 1e3
        self.r_diag: np.ndarray = np.array([1., 1., 1.]) * 1e1

    def get_sampling_time(self) -> float:
        return self.tf / self.n


class SafetyFilter3Dof:
    """
    Safety filter
    """

    def __init__(self,
                 model: "SymbolicFlexibleArm3DOF",
                 model_nonsymbolic,
                 x0: np.ndarray,
                 x0_ee: np.ndarray,
                 options: SafetyFilter3dofOptions):

        self.u_max = 1e6
        self.model_ns = model_nonsymbolic
        self.fa_model = model
        model, constraint_expr = model.get_acados_model_safety()
        self.model = model
        self.options = options
        self.debug_timings = []
        self.debug_total_timings = []
        self.iteration_counter = 0
        self.inter_t2q = None
        self.inter_t2dq = None
        self.inter_pee = None

        # set up simulator for initial state
        integrator = 'RK45'
        self.offline_controller = OfflineController()
        self.sim = Simulator(self.model_ns, self.offline_controller, integrator, None)

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
        ocp.model.con_h_expr = constraint_expr
        # get constraint parameters
        self.n_constraints = constraint_expr.shape[0]
        ns = self.n_constraints  # We just constrain the constraints in the constraint-expression
        nsh = self.n_constraints
        self.current_slacks = np.zeros((ns,))
        ocp.cost.zl = np.array([1, 1, 1])
        ocp.cost.Zl = np.array([1e1, 1e1, 1e1]) * 1e5
        ocp.cost.zu = ocp.cost.zl
        ocp.cost.Zu = ocp.cost.Zl
        ocp.constraints.lh = -np.ones((self.n_constraints,)) * 1e3
        ocp.constraints.lh[0] = 0.
        ocp.constraints.uh = np.ones((self.n_constraints,)) * 1e3
        ocp.constraints.lsh = np.zeros(nsh)
        ocp.constraints.ush = np.zeros(nsh)
        ocp.constraints.idxsh = np.array(range(self.n_constraints))

        # some checks
        assert (nx == options.q_diag.shape[0] == options.q_e_diag.shape[0])
        assert (nu == options.r_diag.shape[0])
        assert (nz == options.z_diag.shape[0] == options.z_e_diag.shape[0])

        # set dimensions
        ocp.dims.N = options.n

        # set cost module
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'

        options.q_diag[0:int(len(options.q_diag) / 2)] = 0
        options.q_diag[int(len(options.q_diag) / 2):] = 1
        options.q_e_diag[0:int(len(options.q_e_diag) / 2)] = 0
        options.q_e_diag[int(len(options.q_e_diag) / 2):] = 1e5
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
        self.x0_ee = x0_ee
        # set constraints
        umax = self.u_max * np.ones((nu,))

        ocp.constraints.constr_type = 'BGH'
        ocp.constraints.lbu = -umax
        ocp.constraints.ubu = umax
        ocp.constraints.x0 = x0.reshape((nx,))
        ocp.constraints.idxbu = np.array(range(nu))
        #ocp.constraints.idxbx = np.array([0])
        #ocp.constraints.lbx = -np.array([np.pi / 2])
        #ocp.constraints.ubx = np.array([np.pi / 2])

        # solver options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'  # FULL_CONDENSING_QPOASES
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'IRK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'  # SQP_RTI, SQP
        ocp.solver_options.nlp_solver_max_iter = options.nlp_iter

        ocp.solver_options.sim_method_num_stages = 2
        ocp.solver_options.sim_method_num_steps = 2
        ocp.solver_options.qp_solver_cond_N = int(1 * options.n)

        # set prediction horizon
        ocp.solver_options.tf = options.tf

        self.acados_ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp_' + model.name + '.json')

        # set costs only for the first stage, ignore the rest
        for stage in range(self.options.n):
            if stage == 0:
                self.acados_ocp_solver.cost_set(stage, "W", scipy.linalg.block_diag(Q,
                                                                                    R,
                                                                                    np.zeros_like(Z)))
            else:
                self.acados_ocp_solver.cost_set(stage, "W", scipy.linalg.block_diag(Q,
                                                                                    R * 1e-1,
                                                                                    np.zeros_like(Z)))
        self.acados_ocp_solver.cost_set(self.options.n, "W", scipy.linalg.block_diag(Q_e, np.zeros_like(Z)))

    def reset(self):
        self.debug_timings = []
        self.iteration_counter = 0

    def filter(self, q: np.ndarray, dq: np.ndarray, u0: np.ndarray):
        # set initial state
        start_time = time.time()
        xcurrent = np.vstack((q, dq))
        self.acados_ocp_solver.set(0, "lbx", xcurrent)
        self.acados_ocp_solver.set(0, "ubx", xcurrent)

        # simulate states
        self.sim.controller.set_u(u0)
        # Todo: This initialisation takes forever. Try without.
        """
        x_sim, u_sim, x_hat_ = self.sim.simulate(xcurrent.flatten(), self.options.get_sampling_time(), self.options.n)
        #t = np.arange(0, n_iter + 1) * ts

        for i in range(self.options.n + 1):
            self.acados_ocp_solver.set(i, "x", x_sim[i,:])
        for i in range(self.options.n):
            self.acados_ocp_solver.set(i, "u", u_sim[i, :])
        # for i in range(self.options.n):
        #    self.acados_ocp_solver.set(0, "z", self.x0_ee)
        """
        # set u0 reference
        stage = 0
        yref = np.vstack((np.zeros((self.nx, 1)), u0.transpose(), np.zeros((self.nz, 1)))).flatten()
        self.acados_ocp_solver.cost_set(stage, "yref", yref)

        # acados solve NLP
        status = self.acados_ocp_solver.solve()

        # Get timing result
        self.debug_timings.append(self.acados_ocp_solver.get_stats("time_tot")[0])

        # Check for errors in acados
        if status != 0:
            raise Exception('acados returned status {} in time step {}. Exiting.'.format(status,
                                                                                         self.iteration_counter))
        self.iteration_counter += 1

        # Retrieve control u
        u_output = self.acados_ocp_solver.get(0, "u")
        #print("delta u: {}".format(u0 - u_output))

        self.debug_total_timings.append(time.time() - start_time)
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

        def compute_torques(self, q, dq):
            u = super(ControllerSafetyWrapper, self).compute_torques(q, dq)
            u_safe = safety_filter.filter(u0=u, q=q, dq=dq)
            return u_safe

    return ControllerSafetyWrapper
