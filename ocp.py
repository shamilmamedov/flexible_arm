import numpy as np
import casadi as cs
from dataclasses import dataclass

from  flexible_arm_3dof import (SymbolicFlexibleArm3DOF, get_rest_configuration)
from animation import Panda3dAnimator
from poly5_planner import Poly5Trajectory
import plotting


class OCPOptions:
    def __init__(self, dt: float, tf: float, n_seg: int) -> None:
        nq = 1 + 2 * (n_seg + 1)

        self.dt = dt
        self.tf = tf
        self.N = int(tf/dt)
        self.Q = np.diag([*[0.1]*nq, *[0.1]*nq])
        self.Qe = np.diag([*[0]*nq, *[10]*nq])
        self.R = np.diag([0.01, 0.001, 0.01])
        self.V = 1e+3*np.diag([1., 1.2, 1.])
        self.Ve = 1e+3*np.diag([1] * 3)
        self.solver = 'ipopt'


class OptimalControlProblem:
    def __init__(self, model: SymbolicFlexibleArm3DOF, 
                 pee_ref: np.ndarray, opts: OCPOptions) -> None:
        self.model = model
        self.pee_ref = pee_ref
        self.opts = opts

        # Formulate OCP and define solver
        self.formulate_ocp()
        self.define_solver()

    def formulate_ocp(self) -> None:
        N = self.opts.N
        pee_ref = self.pee_ref

        # OCP formulation using Opti
        opti = cs.Opti()

        # Symbolic variables
        x_sym = opti.variable(N+1, self.model.nx)
        u_sym = opti.variable(N, self.model.nu)


        # Constraints
        for k in range(N):
            # Multiple shooting constraints
            x_next = self.model.F(x_sym[k,:], u_sym[k,:])
            opti.subject_to(x_next == x_sym[k+1,:].T)

            # Path constraints
            dqak = x_sym[k, self.model.dqa_idx].T
            opti.subject_to(opti.bounded(-self.model.dqa_max, 
                            dqak, self.model.dqa_max))

            opti.subject_to(opti.bounded(-self.model.tau_max, 
                            u_sym[k,:].T, self.model.tau_max))

        # Boundary constraints
        q_t0 = x_sym[0, :self.model.nq]
        dq_t0 = x_sym[0, self.model.nq:]
        pee_t0 = self.model.p_ee(q_t0)
        opti.subject_to(pee_t0 == pee_ref[0,:])
        opti.subject_to(dq_t0.T == np.zeros(self.model.nq))

        q_tf = x_sym[-1, :self.model.nq]
        pee_tf = self.model.p_ee(q_tf)
        opti.subject_to(pee_tf == pee_ref[-1,:])


        # Objective
        # Stage cost
        objective = 0 
        for k in range(N):
            pee_k = self.model.p_ee(x_sym[k, :self.model.nq])
            objective += u_sym[k,:] @ self.opts.R @ u_sym[k,:].T + \
                        (pee_k - pee_ref[k,:]).T @ self.opts.V @ (pee_k - pee_ref[k,:])
                            # (x_sym[k,:].T - x_t0).T @ Q @ (x_sym[k,:].T - x_t0) + \
        # Terminal cost
        # TODO Increase the cost on final ee position
        objective += (pee_tf - pee_ref[-1,:]).T @ self.opts.Ve @ (pee_tf - pee_ref[-1,:]) + \
                     x_sym[-1,:] @ self.opts.Qe @ x_sym[-1,:].T

        opti.minimize(objective)

        self.__opti = opti
        self.__x_sym = x_sym
        self.__u_sym = u_sym

    def define_solver(self) -> None:
        # Define a solver
        p_opts = {"expand": False} # plugin options
        s_opts = {'max_iter': 250, 'print_level': 1, # 'hessian_approximation':'limited-memory',
                    'linear_solver':'mumps'} # solver options
        self.__opti.solver(self.opts.solver, p_opts, s_opts)

    def solve(self, q_t0_guess):
        # Initial guess for the solver
        N = self.opts.N
        x_t0 = np.vstack((q_t0_guess, np.zeros_like(q_t0_guess)))
        x_guess = np.repeat(x_t0.T, N+1, axis=0)
        u_guess = np.repeat(self.model.gravity_torque(q_t0_guess).T, N, axis=0)

        self.__opti.set_initial(self.__x_sym, x_guess)
        self.__opti.set_initial(self.__u_sym, u_guess)

        # Solve OCP
        sol = self.__opti.solve()

        # Parse optimal solution
        u_opt = sol.value(self.__u_sym)
        x_opt = sol.value(self.__x_sym)
        q_opt = x_opt[:, :self.model.nq]
        dq_opt = x_opt[:, self.model.nq:]
        t_opt = np.arange(0, N+1, 1)*self.opts.dt

        return t_opt, q_opt, dq_opt, u_opt


if __name__ == "__main__":
    # Trajectory parameters
    # Initial state of the real system (simulation model)
    n_seg_ocp = 0
    qa_t0 = np.array([np.pi/2, np.pi/10, -np.pi/8])
    q_t0 = get_rest_configuration(qa_t0, n_seg_ocp)
    dq_t0 = np.zeros_like(q_t0)
    x_t0 = np.vstack((q_t0, dq_t0))

    # Compute reference ee position
    qa_tf = np.array([0., 2*np.pi/5, -np.pi/3])
    q_tf = get_rest_configuration(qa_tf, n_seg_ocp)
    dq_tf = np.zeros_like(q_tf)
    x_tf = np.vstack((q_tf, dq_tf))

    # Trajectory and model parameters/opts
    tf = 0.75
    dt = 0.01

    # Model of the robot
    model = SymbolicFlexibleArm3DOF(n_seg=n_seg_ocp, dt=dt,
                                    integrator='cvodes')

    # Compute reference trajectory by Poly5planner
    pee_t0 = np.array(model.p_ee(q_t0))
    pee_tf = np.array(model.p_ee(q_tf))
    pee_ref = Poly5Trajectory(pee_t0, pee_tf, tf, dt).design_traj()[0]

    # Design and solve OCP
    opts = OCPOptions(dt, tf, n_seg_ocp)
    ocp = OptimalControlProblem(model, pee_ref, opts)
    t_opt, q_opt, dq_opt, u_opt = ocp.solve(q_t0)

    # Compute forward kinematics
    N = opts.N
    pee_opt = np.zeros((N+1, 3))
    for k in range(N+1):
        pee_opt[k,:] = np.array(model.p_ee(q_opt[k,:])).flatten()

    # Plot optimnal trajectories
    plotting.plot_ee_positions(t_opt, pee_opt, pee_ref)
    # plotting.plot_joint_positions(t_opt, q_opt, n_seg_ocp)
    plotting.plot_joint_velocities(t_opt, dq_opt, n_seg_ocp)
    plotting.plot_controls(t_opt[:-1], u_opt)

    # Visualize optimal motion
    animator = Panda3dAnimator(model.urdf_path, dt, q_opt).play(2)