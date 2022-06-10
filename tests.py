#!/usr/bin/env python3

import numpy as np
import casadi as cs
import pinocchio as pin

from flexible_arm import FlexibleArm, SymbolicFlexibleArm

def test_casadi_aba():
    """ Test a casadi function for computing forward kinematics
    using Articulated Rigid Body Algorithm (aba)
    """
    # Load casadi function for evaluating aba
    casadi_aba = cs.Function.load('models/aba.casadi')

    # Load flexible arm instance
    arm = FlexibleArm()

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
    sym_arm = SymbolicFlexibleArm()
    num_arm = FlexibleArm()

    # Compute and compare ode in a loop
    for _ in range(15):
        q = -np.pi + 2*np.pi*np.random.rand(num_arm.nq,1)
        dq = -2*np.pi + 4*np.pi*np.random.rand(num_arm.nq,1)
        x = np.vstack((q, dq))
        tau = -25 + 50*np.random.rand(num_arm.nu,1)
        num_ode = num_arm.ode(x, tau)
        sym_ode = np.array(sym_arm.ode(x, tau))
        np.testing.assert_allclose(num_ode, sym_ode)

def test_casadi_forward_simulation():
    """ Test casadi model based forward simulation of the flexible
    arm robot by comparing it to a forward simulation using 
    numerical pinocchio based model
    """
    pass

