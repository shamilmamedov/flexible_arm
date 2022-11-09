import matplotlib
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from flexible_arm_3dof import (FlexibleArm3DOF, SymbolicFlexibleArm3DOF, 
                                get_rest_configuration)
from animation import Animator, Panda3dAnimator
from controller import DummyController, PDController3Dof
from simulation import Simulator, SimulatorOptions
from estimator import ExtendedKalmanFilter
import plotting

if __name__ == "__main__":
    # Simulation parametes
    dt = 0.01
    n_iter = 300

    # Create FlexibleArm instance
    n_seg_sim = 10
    fa = SymbolicFlexibleArm3DOF(n_seg_sim)

    # Initial state
    qa = np.zeros(3)
    q = get_rest_configuration(qa, n_seg_sim)
    dq = np.zeros_like(q)
    x0 = np.concatenate((q,dq)).reshape(-1,1)

    # Reference
    qa_ref = np.array([2, 0.5, 1])
    q_ref = get_rest_configuration(qa_ref, n_seg_sim)

    # Estimator
    # E = None
    n_seg_est = 3
    est_model = SymbolicFlexibleArm3DOF(n_seg_est, dt=dt)
    p0_q, p0_dq = [0.05] * est_model.nq, [1e-3] * est_model.nq
    P0 = np.diag([*p0_q, *p0_dq])
    q_q = [1e-4, *[1e-3] * (est_model.nq - 1)]
    q_dq = [1e-1, *[5e-1] * (est_model.nq - 1)]
    Q = np.diag([*q_q, *q_dq])
    r_q, r_dq, r_pee = [3e-5] * 3, [5e-2] * 3, [1e-3] * 3
    R = 10*np.diag([*r_q, *r_dq, *r_pee])

    q_est = get_rest_configuration(qa, n_seg_est)
    dq_est = np.zeros_like(q_est)
    x0_est = np.concatenate((q_est, dq_est)).reshape(-1,1)
    E = ExtendedKalmanFilter(est_model, x0_est, P0, Q, R)

    # Controller
    # controller = DummyController()
    C = PDController3Dof(Kp=(40, 4, 2), Kd=(1.5, 0.1, 0.05),
                         n_seg=n_seg_est, q_ref=q_ref)

    # Simulate
    opts = SimulatorOptions(contr_input_states='real') # 'real', 'estimated'
    integrator = 'cvodes'
    sim = Simulator(fa, C, integrator, E, opts)
    x, u, y, x_hat = sim.simulate(x0.flatten(), n_iter)
    t = np.arange(0, n_iter + 1) * dt

    # Parse joint positions and plot active joints positions
    n_skip = 1
    q = x[::n_skip, :fa.nq]

    # plot_real_states_vs_estimate(t, x, x_hat)
    # plot_joint_positions(t[::10], q, n_seg, q_ref)
    # plot_controls(t[:-1], u)
    # plotting.plot_measurements(t, y)
    plotting.plot_real_states_vs_estimate(t, x, x_hat, n_seg_sim, n_seg_est)
    # plot_joint_positions(t[::n_skip], q, n_seg, q_ref)
    
    # Animate simulated motion
    animator = Panda3dAnimator(fa.urdf_path, 0.01, q).play(3)

