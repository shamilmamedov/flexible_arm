import os
import numpy as np
import casadi as cs
import pinocchio as pin

from flexible_arm_3dof import FlexibleArm3DOF, SymbolicFlexibleArm3DOF
from estimator import ExtendedKalmanFilter

n_segs = [3, 5, 10]
n_seg_int2str = {1:'one', 3:'three', 5:'five', 10:'ten'}
robot_folder = 'models/three_dof/'

def test_casadi_aba():
    """ Test a casadi function for computing forward kinematics
    using Articulated Rigid Body Algorithm (aba)
    """
    # Path to a folder with model description
    for n_seg in n_segs:
        model_folder = os.path.join(robot_folder, n_seg_int2str[n_seg] + '_segments/')

        # Load casadi function for evaluating aba
        casadi_aba = cs.Function.load(os.path.join(model_folder, 'aba.casadi'))

        # Load flexible arm instance
        arm = FlexibleArm3DOF(n_seg)

        for _ in range(15):
            # Random position, velocity and torques
            q = -np.pi + 2*np.pi*np.random.rand(arm.nq,1)
            dq = -2*np.pi + 4*np.pi*np.random.rand(arm.nq,1)
            tau = -15 + 30*np.random.rand(arm.nq,1)

            # Compute acclerations and compare
            ddq_casadi = np.array(casadi_aba(q, dq, tau))
            ddq_num = pin.aba(arm.model, arm.data, q, dq, tau).reshape(-1,1) 
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
            q = -np.pi + 2*np.pi*np.random.rand(num_arm.nq,1)
            dq = -2*np.pi + 4*np.pi*np.random.rand(num_arm.nq,1)
            x = np.vstack((q, dq))
            tau = -25 + 50*np.random.rand(num_arm.nu,1)
            num_ode = num_arm.ode(x, tau)
            sym_ode = np.array(sym_arm.ode(x, tau))
            np.testing.assert_allclose(num_ode, sym_ode)

def test_casadi_fk():
    """ Tests a casadi function for computing forward kinematics by
    comparing it to the numerical pinocchio values
    """
    # Path to a folder with model description
    for n_seg in n_segs:
        model_folder = os.path.join(robot_folder, n_seg_int2str[n_seg] + '_segments/')

        # Load casadi function for evaluating forward kinematics
        casadi_fkp = cs.Function.load(os.path.join(model_folder, 'fkp.casadi'))

        # Load flexible arm instance 
        arm = FlexibleArm3DOF(n_seg)

        for _ in range(15):
            # Random position, velocity and torques
            q = -np.pi + 2*np.pi*np.random.rand(arm.nq,1)

            # Compute acclerations and compare
            pee_casadi = np.array(casadi_fkp(q))
            pee_num = arm.fk_ee(q)[1] 
            np.testing.assert_allclose(pee_casadi, pee_num)

def test_casadi_vee():
    """ Tests a casadi function for computing ee velocity
    """
    for n_seg in n_segs:
        model_folder = os.path.join(robot_folder, n_seg_int2str[n_seg] + '_segments/')

        # Load casadi function for evaluating forward kinematics
        casadi_fkp = cs.Function.load(os.path.join(model_folder, 'fkv.casadi'))

        # Load flexible arm instance 
        arm = FlexibleArm3DOF(n_seg)

        for _ in range(15):
            # Random position, velocity and torques
            q = -np.pi + 2*np.pi*np.random.rand(arm.nq,1)
            dq = -2*np.pi + 4*np.pi*np.random.rand(arm.nq,1)

            # Compute acclerations and compare
            vee_casadi = np.array(casadi_fkp(q, dq))
            vee_num = arm.ee_velocity(q, dq)[:3,:] 
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
    print(x_hat)


if __name__ == "__main__":
    test_casadi_aba()
    test_casadi_ode()
    test_casadi_fk()
    test_casadi_vee()
    test_SymbolicFlexibleArm()
    test_EKF()