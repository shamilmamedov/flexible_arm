import os
import time
from matplotlib import pyplot as plt
import numpy as np
import casadi as cs
import pinocchio as pin
import pandas as pd

from envs.flexible_arm_3dof import (FlexibleArm3DOF, SymbolicFlexibleArm3DOF, 
                               get_rest_configuration, inverse_kinematics_rb)
from estimator import ExtendedKalmanFilter
from simulation import Simulator, SimulatorOptions
from controller import DummyController, PDController3Dof, FeedforwardController


n_segs = [0, 1, 2, 3, 5, 10]
n_seg_int2str = {0: 'zero', 1:'one', 2:'two', 3: 'three', 5: 'five', 10: 'ten'}
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
    """ Test Extended Kalman Filter implementation
    """
    # Create a model of the robot
    est_model = SymbolicFlexibleArm3DOF(3, integrator='collocation')

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

    # Update the estimate of the state in the beginning when
    # there is no input yet
    n_iter = 100
    t0 = time.time()
    for _ in range(n_iter):
        x_hat = E.estimate(y)
    tf = time.time()
    print(f"Average update time: {1000*(tf-t0)/n_iter:.4f} ms")

    u = np.zeros((3,1))
    t0 = time.time()
    for _ in range(n_iter):
        x_hat = E.estimate(y, u)
    tf = time.time()
    print(f"Average predict-update time: {1000*(tf-t0)/n_iter:.4f} ms")



def test_rest_configuration_computation():
    """ Test a function for computing the rest position from active
    joint configurations
    """
    for n_seg in n_segs:
        nq = 3 + 2*n_seg
        idxs = set(np.arange(0, nq))
        idxs_a = {0, 1, 2 + n_seg}
        idxs_p = idxs.difference(idxs_a)

        # Tests that when there is no gravity then the deformations
        # of the passive joints are zero
        if n_seg > 0:
            qa = np.array([0, np.pi/2, 0])
            q = get_rest_configuration(qa, n_seg)
            qp = q[list(idxs_p)]
            np.testing.assert_array_equal(qp, np.zeros((2*n_seg,1)))

        qa = np.random.randn(3)
        q = get_rest_configuration(qa, n_seg)


def test_inverse_kinematics():
    """
    TODO it is difficult to test because the IK can
    come up with any of the three configurations
    """
    model_folder = 'models/three_dof/zero_segments/'
    fkp = cs.Function.load(model_folder + 'fkp.casadi')

    for k in range(2):
        q = -np.pi + 2*np.pi*np.random.uniform(size=(3,))
        pee = np.array(fkp(q))
        q_ik = inverse_kinematics_rb(pee)
        print(f'iter {k}')


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
    dt = 0.001
    n_iter = 100

    # Models
    n_seg = 10
    model = SymbolicFlexibleArm3DOF(n_seg, dt=dt)

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
    # C = PDController3Dof(Kp, Kd, n_seg, q_ref)
    C = DummyController()

    # Simulators
    intg1 = 'LSODA'
    intg2 = 'RK45'
    intg3 = 'collocation'
    intg4 = 'cvodes'

    sim_opts = SimulatorOptions(dt=dt, n_iter=n_iter)

    S1 = Simulator(model, C, intg1, E)
    S2 = Simulator(model, C, intg2, E)
    S3 = Simulator(model, C, intg3, E)
    S4 = Simulator(model, C, intg4, E)

    # Simulate the rigid body approxmiation
    t0_LSODA = time.time()
    x1, u1, y1, _ = S1.simulate(x0.flatten())
    tf_LSODA = time.time()

    t0_RK45 = time.time()
    x2, u2, y2, _ = S2.simulate(x0.flatten())
    tf_RK45 = time.time()

    t0_clc = time.time()
    x3, u3, y3, _ = S3.simulate(x0.flatten())
    tf_clc = time.time()

    t0_cvd = time.time()
    x4, u4, y4, _ = S4.simulate(x0.flatten())
    tf_cvd = time.time()
    
    t = np.arange(0, n_iter + 1) * dt

    print('Execution time LSODA:', tf_LSODA-t0_LSODA)
    print('Execution time RK45:', tf_RK45-t0_RK45)
    print('Execution time collocation:', tf_clc-t0_clc)
    print('Execution time cvodes:', tf_cvd-t0_cvd)

    # Plot outputs
    y_lbls = [r'$\mathrm{ee}_x$ [m]', r'$\mathrm{ee}_y$ [m]',
              r'$\mathrm{ee}_z$ [m]']
    legends = ['LSODA', 'RK45', 'collocation', 'cvodes']
    _, axs = plt.subplots(3, 1, sharex=True, figsize=(6, 7))
    for k, ax in enumerate(axs.reshape(-1)):
        ax.plot(t, y1[:, 6+k], label=legends[0])
        ax.plot(t, y2[:, 6+k], label=legends[1])
        ax.plot(t, y3[:, 6+k], label=legends[2])
        ax.plot(t, y4[:, 6+k], label=legends[3])
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
    test_rest_configuration_computation()
    test_inverse_kinematics()
    # compare_different_discretization()
    # compare_discretized_num_sym_models()
    compare_different_integrators()