#!/usr/bin/env python3

import os
import numpy as np
import casadi as cs
import pinocchio as pin

from flexible_arm import FlexibleArm, SymbolicFlexibleArm

def test_casadi_aba():
    """ Test a casadi function for computing forward kinematics
    using Articulated Rigid Body Algorithm (aba)
    """
    n_segs = [3, 5, 10]

    # Path to a folder with model description
    n_seg_int2str = {1:'one', 3:'three', 5:'five', 10:'ten'}

    for n_seg in n_segs:
        model_folder = 'models/' + n_seg_int2str[n_seg] + '_segments/'

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
        model_folder = 'models/' + n_seg_int2str[n_seg] + '_segments/'

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
        model_folder = 'models/' + n_seg_int2str[n_seg] + '_segments/'

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

if __name__ == "__main__":
    test_casadi_fk()
    test_casadi_vee()