#!/usr/bin/env python3

import os
import numpy as np
import casadi as cs
import pinocchio as pin
import pandas as pd
import matplotlib.pyplot as plt 

from flexible_arm import FlexibleArm, SymbolicFlexibleArm
from controller import PDController
from simulation import Simulator

def test_casadi_aba():
    """ Test a casadi function for computing forward kinematics
    using Articulated Rigid Body Algorithm (aba)
    """
    n_segs = [3, 5, 10]

    # Path to a folder with model description
    n_seg_int2str = {1:'one', 3:'three', 5:'five', 10:'ten'}

    for n_seg in n_segs: 
        model_folder = 'models/one_dof/' + n_seg_int2str[n_seg] + '_segments/'

        # Load casadi function for evaluating aba
        casadi_aba = cs.Function.load(os.path.join(model_folder, 'aba.casadi'))

        # Load flexible arm instance
        arm = FlexibleArm(n_seg)

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
    n_segs = [3, 5, 10]

    # Instantiate numerial and symbolic arm models
    for n_seg in n_segs: 
        sym_arm = SymbolicFlexibleArm(n_seg)
        num_arm = FlexibleArm(n_seg)

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
    n_segs = [3, 5, 10]

    # Path to a folder with model description
    n_seg_int2str = {1:'one', 3:'three', 5:'five', 10:'ten'}

    for n_seg in n_segs: 
        model_folder = 'models/one_dof/' + n_seg_int2str[n_seg] + '_segments/'

        # Load casadi function for evaluating forward kinematics
        casadi_fkp = cs.Function.load(os.path.join(model_folder, 'fkp.casadi'))

        # Load flexible arm instance 
        arm = FlexibleArm(n_seg)

        for _ in range(15):
            # Random position, velocity and torques
            q = -np.pi + 2*np.pi*np.random.rand(arm.nq,1)

            # Compute acclerations and compare
            pee_casadi = np.array(casadi_fkp(q))
            pee_num = arm.fk_ee(q)[1] 
            np.testing.assert_allclose(pee_casadi, pee_num[[0,2],:])

def test_casadi_vee():
    """ Tests a casadi function for computing ee velocity
    """
    n_segs = [3, 5, 10]

    # Path to a folder with model description
    n_seg_int2str = {1:'one', 3:'three', 5:'five', 10:'ten'} 

    for n_seg in n_segs: 
        model_folder = 'models/one_dof/' + n_seg_int2str[n_seg] + '_segments/'

        # Load casadi function for evaluating forward kinematics
        casadi_fkp = cs.Function.load(os.path.join(model_folder, 'fkv.casadi'))

        # Load flexible arm instance 
        arm = FlexibleArm(n_seg)

        for _ in range(15):
            # Random position, velocity and torques
            q = -np.pi + 2*np.pi*np.random.rand(arm.nq,1)
            dq = -2*np.pi + 4*np.pi*np.random.rand(arm.nq,1)

            # Compute acclerations and compare
            vee_casadi = np.array(casadi_fkp(q, dq))
            vee_num = arm.ee_velocity(q, dq)[:3,:] 
            np.testing.assert_allclose(vee_casadi, vee_num[[0,2],:])

def test_casadi_forward_simulation():
    """ Test casadi model based forward simulation of the flexible
    arm robot by comparing it to a forward simulation using 
    numerical pinocchio based model
    """
    pass

def test_SymbolicFlexibleArm():
    """ Tests Symbolic Flexible Arm class, very basic
    """
    n_segs = [3, 5, 10]

    for n_seg in n_segs:
        sarm = SymbolicFlexibleArm(n_seg)
        q = np.random.rand(sarm.nq)
        dq = np.pi*np.random.rand(sarm.nq)
        pee = sarm.p_ee(q)
        vee = sarm.v_ee(q, dq)


def compare_sim_against_matlab():
    """ Test simulator (integrator) against a matlab simulation
    that uses a multibody model built from a URDF with flexbility
    parameters that were inserted manually. 
    NOTE flexibility parameters in the matlab model were hardcoded
            K = [100., 100.] and D = [5., 5.]. For comparison set
            the same parameters in your simulation
    """
    # Create FlexibleArm instance
    n_seg = 3
    fa = FlexibleArm(n_seg)

    # Flexibility parameter checks
    np.testing.assert_equal(np.diag(fa.K), np.array([100., 100.]))
    np.testing.assert_equal(np.diag(fa.D), np.array([5., 5.]))

    # Initial state for simulation
    q = np.zeros((fa.nq, 1))
    dq = np.zeros_like(q)
    x0 = np.vstack((q, dq))

    # Controller
    controller = PDController(Kp=150, Kd=5, q_ref=np.array([np.pi/4]))

    # Simulator params
    ts = 0.01
    n_iter = 100

    sim = Simulator(fa, controller, 'rk45')
    x, u = sim.simulate(x0.flatten(), ts, n_iter)
    t = np.arange(0, n_iter+1)*ts

    # Parse joint positions
    q = x[:,:fa.nq]
    dq = x[:,fa.nq:]

    # Load a simulatio results from matlab
    df = pd.read_csv('data/matlab_sim_3s.csv')
    t_matlab = df['t'].to_numpy()
    q_cols = [f'q{x+1}' for x in range(3)]
    dq_cols = [f'dq{x+1}' for x in range(3)]
    q_matlab = df[q_cols].to_numpy(dtype=np.double)
    dq_matlab = df[dq_cols].to_numpy(dtype=np.double)
    tau_matlab = df['tau'].to_numpy(dtype=np.double).reshape(-1,1)

    print(f"|tau_ours - tau_matlab| = {np.linalg.norm(u - tau_matlab[:n_iter,:]):.5f}")
    print(f"|q_ours - q_matlab| = {np.linalg.norm(q - q_matlab):.5f}")
    print(f"|dq_ours - dq_matlab| = {np.linalg.norm(dq - dq_matlab):.5f}")

    # Plot matlab and our simulation results against each other
    _, ax_tau = plt.subplots()
    ax_tau.plot(t[:-1], u, lw=2, label='ours')
    ax_tau.plot(t_matlab, tau_matlab, '--', lw=2, label='matlab')
    ax_tau.legend()
    ax_tau.set_xlabel(r'$t$ (s)')
    ax_tau.set_ylabel(r'$\tau$ (Nm)')
    plt.tight_layout()

    _, ax_qp = plt.subplots()
    ax_qp.plot(t, q[:,1], lw=2, label='qp1 ours')
    ax_qp.plot(t, q[:,2], lw=2, label='qp2 ours')
    ax_qp.plot(t_matlab, q_matlab[:,1], '--', lw=2, label='qp1 matlab')
    ax_qp.plot(t_matlab, q_matlab[:,2], '--', lw=2, label='qp2 matlab')
    ax_qp.legend()
    ax_qp.set_xlabel(r'$t$ (s)')
    ax_qp.set_ylabel(r'$q_p$ (rad)')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_sim_against_matlab()