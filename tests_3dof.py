import os
import time
from matplotlib import pyplot as plt
import numpy as np
import casadi as cs
import pinocchio as pin
import pandas as pd

from flexible_arm_3dof import FlexibleArm3DOF, SymbolicFlexibleArm3DOF
from estimator import ExtendedKalmanFilter
from simulation import Simulator
from controller import DummyController, PDController3Dof, FeedforwardController

n_segs = [2, 3, 5, 10]
n_seg_int2str = {2: 'two', 3: 'three', 5: 'five', 10: 'ten'}
robot_folder = 'models/three_dof/'


def test_casadi_aba():
    """ Test a casadi function for computing forward kinematics
    using Articulated Rigid Body Algorithm (aba)
    """
    # Path to a folder with model description
    for n_seg in n_segs:
        model_folder = os.path.join(
            robot_folder, n_seg_int2str[n_seg] + '_segments/')

        # Load casadi function for evaluating aba
        casadi_aba = cs.Function.load(os.path.join(model_folder, 'aba.casadi'))

        # Load flexible arm instance
        arm = FlexibleArm3DOF(n_seg)

        for _ in range(15):
            # Random position, velocity and torques
            q = -np.pi + 2*np.pi*np.random.rand(arm.nq, 1)
            dq = -2*np.pi + 4*np.pi*np.random.rand(arm.nq, 1)
            tau = -15 + 30*np.random.rand(arm.nq, 1)

            # Compute acclerations and compare
            ddq_casadi = np.array(casadi_aba(q, dq, tau))
            ddq_num = pin.aba(arm.model, arm.data, q, dq, tau).reshape(-1, 1)
            np.testing.assert_allclose(ddq_casadi, ddq_num)


def test_casadi_ode():
    """ Tests symbolic casadi ode model of the robot by comparing it
    to numerical ode built using pinocchio
    """
    # Instantiate numerial and symbolic arm models
    for n_seg in n_segs:
        sym_arm = SymbolicFlexibleArm3DOF(n_seg)
        num_arm = FlexibleArm3DOF(n_seg)

        # Compute and compare ode in a loop
        for _ in range(15):
            q = -np.pi + 2*np.pi*np.random.rand(num_arm.nq, 1)
            dq = -2*np.pi + 4*np.pi*np.random.rand(num_arm.nq, 1)
            x = np.vstack((q, dq))
            tau = -25 + 50*np.random.rand(num_arm.nu, 1)
            num_ode = num_arm.ode(x, tau)
            sym_ode = np.array(sym_arm.ode(x, tau))
            np.testing.assert_allclose(num_ode, sym_ode)


def test_casadi_fk():
    """ Tests a casadi function for computing forward kinematics by
    comparing it to the numerical pinocchio values
    """
    # Path to a folder with model description
    for n_seg in n_segs:
        model_folder = os.path.join(
            robot_folder, n_seg_int2str[n_seg] + '_segments/')

        # Load casadi function for evaluating forward kinematics
        casadi_fkp = cs.Function.load(os.path.join(model_folder, 'fkp.casadi'))

        # Load flexible arm instance
        arm = FlexibleArm3DOF(n_seg)

        for _ in range(15):
            # Random position, velocity and torques
            q = -np.pi + 2*np.pi*np.random.rand(arm.nq, 1)

            # Compute acclerations and compare
            pee_casadi = np.array(casadi_fkp(q))
            pee_num = arm.fk_ee(q)[1]
            np.testing.assert_allclose(pee_casadi, pee_num)


def test_casadi_vee():
    """ Tests a casadi function for computing ee velocity
    """
    for n_seg in n_segs:
        model_folder = os.path.join(
            robot_folder, n_seg_int2str[n_seg] + '_segments/')

        # Load casadi function for evaluating forward kinematics
        casadi_fkp = cs.Function.load(os.path.join(model_folder, 'fkv.casadi'))

        # Load flexible arm instance
        arm = FlexibleArm3DOF(n_seg)

        for _ in range(15):
            # Random position, velocity and torques
            q = -np.pi + 2*np.pi*np.random.rand(arm.nq, 1)
            dq = -2*np.pi + 4*np.pi*np.random.rand(arm.nq, 1)

            # Compute acclerations and compare
            vee_casadi = np.array(casadi_fkp(q, dq))
            vee_num = arm.ee_velocity(q, dq)[:3, :]
            np.testing.assert_allclose(vee_casadi, vee_num)


def test_casadi_forward_simulation():
    """ Test casadi model based forward simulation of the flexible
    arm robot by comparing it to a forward simulation using 
    numerical pinocchio based model
    """
    pass


def test_SymbolicFlexibleArm():
    """ Tests Symbolic Flexible Arm class, very basic. Make sure
    that all functions are collable and do not crash
    """
    for n_seg in n_segs:
        sarm = SymbolicFlexibleArm3DOF(n_seg)
        q = np.random.rand(sarm.nq)
        dq = np.pi*np.random.rand(sarm.nq)
        pee = np.array(sarm.p_ee(q))
        vee = np.array(sarm.v_ee(q, dq))

        x = cs.vertcat(q, dq)
        u = np.random.rand(sarm.nu)
        x_next = sarm.F(x, u)
        h = np.array(sarm.h(x))
        dh_dx = np.array(sarm.dh_dx(x))
        df_dx = np.array(sarm.df_dx(x, u))
        df_du = np.array(sarm.df_du(x, u))
        dF_dx = np.array(sarm.dF_dx(x, u))
        # print(f'p_ee = {pee.flatten()}')
        # print(f'v_ee = {vee.flatten()}')
        # print(f'h = {h.flatten()}')
        # print(f'dy_dx = {dh_dx}')


def test_EKF():
    # Create a model of the robot
    est_model = SymbolicFlexibleArm3DOF(3)

    # Initial state
    q = np.zeros((est_model.nq, 1))
    dq = np.zeros_like(q)
    x0 = np.vstack((q, dq))

    # Parameters of the estimator
    P0 = 0.01*np.ones((est_model.nx, est_model.nx))
    q_q, q_dq = [1e-2]*est_model.nq, [1e-1]*est_model.nq
    Q = np.diag([*q_q, *q_dq])
    r_q, r_dq, r_pee = [3e-4]*3, [6e-2]*3, [1e-2]*3
    R = np.diag([*r_q, *r_dq, *r_pee])

    # Instantiate estimator
    E = ExtendedKalmanFilter(est_model, x0, P0, Q, R)

    # A numerical model
    sim_model = FlexibleArm3DOF(3)
    delta_q = 1e-2*np.ones((est_model.nq, 1))
    delta_dq = 1e-1*np.ones((est_model.nq, 1))
    delta_x = np.vstack((delta_q, delta_dq))
    y = sim_model.output(x0 + delta_x)

    # Update the estimate of the state
    x_hat = E.estimate(y)
    # print(x_hat)


def compare_different_discretization():
    """ Compares models with different discretization -- starting 
    from a rigid body model and until a fine grid of 10 discretization
    points -- in simulatoin. For every simulation the same integrator, 
    controller and initial state are chosen.   
    """
    # Simulation parametes
    ts = 0.001
    n_iter = 750

    # Create FlexibleArm instance
    n_segs = [0, 1, 2, 3, 5, 10]
    fas = [FlexibleArm3DOF(n_seg) for n_seg in n_segs]

    # Initial state; the same for all models
    Q = [np.zeros((m.nq, 1)) for m in fas]
    dQ = [np.zeros((m.nq, 1)) for m in fas]
    X0 = [np.vstack((q, dq)) for q, dq in zip(Q, dQ)]

    # Reference; the same for all models but because of
    Q_ref = [np.zeros((m.nq, 1)) for m in fas]
    for q_ref, n_seg in zip(Q_ref, n_segs):
        q_ref[:2, 0] += [1., 0.5]
        q_ref[1 + n_seg + 1] += 1

    # Estimator
    E = None

    # Controller
    # controller = DummyController()
    Kp = (40, 30, 25)
    Kd = (1.5, 0.25, 0.25)

    # Controller for the rigid body approximation
    C_0s = PDController3Dof(Kp, Kd, n_segs[0], Q_ref[0])

    # Simulators
    integrator = 'LSODA'
    # S = [Simulator(fa, c, integrator, E) for fa, c in zip(fas, C)]
    # Simulator for the rigid body approximation
    S_0s = Simulator(fas[0], C_0s, integrator, E)

    # Simulate the rigid body approxmiation
    x_0s, u_0s, y_0s, _ = S_0s.simulate(X0[0].flatten(), ts, n_iter)
    

    # Create feedforward cotnrollers for RFEM models
    C = [FeedforwardController(u_0s) for _ in n_segs[1:]]
    
    # Create simulators for RFEM models
    S = [Simulator(fa, c, integrator, E) for fa, c in zip(fas[1:], C)]

    # Simulate REFEM models
    x_1s, u_1s, y_1s, _ = S[0].simulate(X0[1].flatten(), ts, n_iter)
    x_2s, u_2s, y_2s, _ = S[1].simulate(X0[2].flatten(), ts, n_iter)
    x_3s, u_3s, y_3s, _ = S[2].simulate(X0[3].flatten(), ts, n_iter)
    x_5s, u_5s, y_5s, _ = S[3].simulate(X0[4].flatten(), ts, n_iter)
    x_10s, u_10s, y_10s, _ = S[4].simulate(X0[5].flatten(), ts, n_iter)

    t = np.arange(0, n_iter + 1) * ts

    # Find the norm of the difference betweem 10s and other models
    delta_0s = y_10s - y_0s
    delta_1s = y_10s - y_1s
    delta_2s = y_10s - y_2s
    delta_3s = y_10s - y_3s
    delta_5s = y_10s - y_5s

    norm_delta_0s = np.linalg.norm(delta_0s, axis=0).reshape(1,-1)
    norm_delta_1s = np.linalg.norm(delta_1s, axis=0).reshape(1,-1)
    norm_delta_2s = np.linalg.norm(delta_2s, axis=0).reshape(1,-1)
    norm_delta_3s = np.linalg.norm(delta_3s, axis=0).reshape(1,-1)
    norm_delta_5s = np.linalg.norm(delta_5s, axis=0).reshape(1,-1)

    norm_delta = np.vstack((norm_delta_0s, norm_delta_1s, norm_delta_2s,
                            norm_delta_3s, norm_delta_5s))

    cols = ['|delta qa_1|', '|delta qa_2|', '|delta qa_3|', 
            '|delta dqa_1|', '|delta dqa_2|', '|delta dqa_3|', 
            '|delta ee_x|', '|delta ee_y|', '|delta ee_z|']

    df = pd.DataFrame(norm_delta, columns=cols, index=[0, 1, 2, 3, 5])
    print(df.iloc[:,6:])


    # Plot outputs
    y_lbls = [r'$\mathrm{ee}_x$ [m]', r'$\mathrm{ee}_y$ [m]',
              r'$\mathrm{ee}_z$ [m]']
    legends = [f'{n_seg}s' for n_seg in n_segs]
    _, axs = plt.subplots(3, 1, sharex=True, figsize=(6, 7))
    for k, ax in enumerate(axs.reshape(-1)):
        ax.plot(t, y_0s[:, 6+k], label=legends[0])
        ax.plot(t, y_1s[:, 6+k], label=legends[1])
        ax.plot(t, y_2s[:, 6+k], label=legends[2])
        ax.plot(t, y_3s[:, 6+k], label=legends[3])
        ax.plot(t, y_5s[:, 6+k], ls='--', label=legends[4])
        ax.plot(t, y_10s[:, 6+k], ls='-.', label=legends[5])
        ax.set_ylabel(y_lbls[k])
        ax.grid(alpha=0.5)
        ax.legend()
    ax.set_xlabel('t [s]')
    plt.tight_layout()

    # Plot controls
    y_lbls = [r'$\tau$ [m]', r'$\tau$ [m]', r'$\tau$ [m]']
    _, axs = plt.subplots(3, 1, sharex=True, figsize=(6, 7))
    for k, ax in enumerate(axs.reshape(-1)):
        ax.plot(t[:-1], u_0s[:, k], label=legends[0])
        ax.plot(t[:-1], u_1s[:, k], label=legends[1])
        ax.plot(t[:-1], u_2s[:, k], label=legends[2])
        ax.plot(t[:-1], u_3s[:, k], label=legends[3])
        ax.plot(t[:-1], u_5s[:, k], ls='--', label=legends[4])
        ax.plot(t[:-1], u_10s[:, k], ls='-.', label=legends[5])
        ax.set_ylabel(y_lbls[k])
        ax.set_xlim([0, 0.025])
        ax.grid(alpha=0.5)
        ax.legend()
    ax.set_xlabel('t [s]')
    plt.tight_layout()

    plt.show()


def compare_discretized_num_sym_models():
    # Simulation parametes
    ts = 0.001
    n_iter = 100

    # Models
    n_seg = 3
    model_num = FlexibleArm3DOF(n_seg)
    model_sym = SymbolicFlexibleArm3DOF(n_seg, ts=ts)

    # Initial states
    q = np.zeros((model_num.nq, 1)) 
    dq = np.zeros((model_num.nq, 1)) 
    x0 = np.vstack((q, dq)) 

    # Reference; the same for all models but because of
    q_ref = np.zeros((model_num.nq, 1)) 
    q_ref[:2, 0] += [1., 0.5]
    q_ref[1 + n_seg + 1] += 1

    # Estimator
    E = None

    # Controller
    Kp = (40, 30, 25)
    Kd = (1.5, 0.25, 0.25)
    C = PDController3Dof(Kp, Kd, n_seg, q_ref)

    # Simulators
    integrator = 'LSODA'
    S_sym = Simulator(model_sym, C, integrator, E)
    S_num = Simulator(model_num, C, integrator, E)

    # Simulate the rigid body approxmiation
    x_sym, u_sym, y_sym, _ = S_sym.simulate(x0.flatten(), ts, n_iter)
    x_num, u_num, y_num, _ = S_num.simulate(x0.flatten(), ts, n_iter)
    t = np.arange(0, n_iter + 1) * ts

    # Some tests and sanity checks
    np.testing.assert_array_almost_equal(y_sym, y_num, decimal=8)

    # Plot outputs
    y_lbls = [r'$\mathrm{ee}_x$ [m]', r'$\mathrm{ee}_y$ [m]',
              r'$\mathrm{ee}_z$ [m]']
    legends = ['sym', 'num']
    _, axs = plt.subplots(3, 1, sharex=True, figsize=(6, 7))
    for k, ax in enumerate(axs.reshape(-1)):
        ax.plot(t, y_sym[:, 6+k], label=legends[0])
        ax.plot(t, y_num[:, 6+k], label=legends[1])
        ax.set_ylabel(y_lbls[k])
        ax.grid(alpha=0.5)
        ax.legend()
    ax.set_xlabel('t [s]')
    plt.tight_layout()
    plt.show()


def compare_different_integrators():
    # Simulation parametes
    ts = 0.001
    n_iter = 100

    # Models
    n_seg = 10
    model = SymbolicFlexibleArm3DOF(n_seg, ts=ts)

    # Initial states
    q = np.zeros((model.nq, 1)) 
    dq = np.zeros((model.nq, 1)) 
    x0 = np.vstack((q, dq)) 

    # Reference; the same for all models but because of
    q_ref = np.zeros((model.nq, 1)) 
    q_ref[:2, 0] += [1., 0.5]
    q_ref[1 + n_seg + 1] += 1

    # Estimator
    E = None

    # Controller
    Kp = (40, 30, 25)
    Kd = (1.5, 0.25, 0.25)
    C = PDController3Dof(Kp, Kd, n_seg, q_ref)

    # Simulators
    intg1 = 'LSODA'
    intg2 = 'RK45'
    intg3 = 'collocation'

    S1 = Simulator(model, C, intg1, E)
    S2 = Simulator(model, C, intg2, E)
    S3 = Simulator(model, C, intg3, E)

    # Simulate the rigid body approxmiation
    t0_LSODA = time.time()
    x1, u1, y1, _ = S1.simulate(x0.flatten(), ts, n_iter)
    tf_LSODA = time.time()

    t0_RK45 = time.time()
    x2, u2, y2, _ = S2.simulate(x0.flatten(), ts, n_iter)
    tf_RK45 = time.time()

    t0_clc = time.time()
    x3, u3, y3, _ = S3.simulate(x0.flatten(), ts, n_iter)
    tf_clc = time.time()
    
    t = np.arange(0, n_iter + 1) * ts

    print('Execution time LSODA:', tf_LSODA-t0_LSODA)
    print('Execution time RK45:', tf_RK45-t0_RK45)
    print('Execution time collocation:', tf_clc-t0_clc)

    # Plot outputs
    y_lbls = [r'$\mathrm{ee}_x$ [m]', r'$\mathrm{ee}_y$ [m]',
              r'$\mathrm{ee}_z$ [m]']
    legends = ['LSODA', 'RK45', 'collocation']
    _, axs = plt.subplots(3, 1, sharex=True, figsize=(6, 7))
    for k, ax in enumerate(axs.reshape(-1)):
        ax.plot(t, y1[:, 6+k], label=legends[0])
        ax.plot(t, y2[:, 6+k], label=legends[1])
        ax.plot(t, y3[:, 6+k], label=legends[2])
        ax.set_ylabel(y_lbls[k])
        ax.grid(alpha=0.5)
        ax.legend()
    ax.set_xlabel('t [s]')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_casadi_aba()
    test_casadi_ode()
    test_casadi_fk()
    test_casadi_vee()
    test_SymbolicFlexibleArm()
    test_EKF()
    # compare_different_discretization()
    # compare_discretized_num_sym_models()
    compare_different_integrators()