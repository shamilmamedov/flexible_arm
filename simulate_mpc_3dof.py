#!/usr/bin/env python3

import numpy as np
import pinocchio as pin
from compute_equilibria import EquilibriaWrapper
from utils import plot_result, print_timings
from flexible_arm_3dof import FlexibleArm3DOF, SymbolicFlexibleArm3DOF
from animation import Animator
from mpc_3dof import Mpc3dofOptions, Mpc3Dof
from simulation import Simulator
from controller import ConstantController, PDController

if __name__ == "__main__":
    # Create FlexibleArm instance
    # Create FlexibleArm instance
    n_seg = 10
    n_seg_mpc = 10
    fa = FlexibleArm3DOF(n_seg)
    fa_sym = SymbolicFlexibleArm3DOF(n_seg_mpc)

    # Sample a random configuration
    q = pin.randomConfiguration(fa.model)
    # fa.visualize(q)

    # Simulate
    q = np.zeros((fa.nq, 1))
    dq = np.zeros_like(q)
    x0 = np.vstack((q, dq))

    q_mpc = np.zeros((fa_sym.nq, 1))
    dq_mpc = np.zeros_like(q_mpc)
    x0_mpc = np.vstack((q_mpc, dq_mpc))

    # Compute an equilibrium in order to find the right set point
    # equi_wrapper = EquilibriaWrapper(model_sym=fa_sym, model=fa, guess_max_torque=0.958)
    u_equi = np.zeros((fa.nu,1))  # the equilibrium is related to a torque value
    # x_equis = equi_wrapper.get_equilibrium_cartesian_states(input_u=u_equi)
    # equi_wrapper.plot()  # plots all equilibria

    mpc_options = Mpc3dofOptions(n_links=n_seg_mpc)
    controller = Mpc3Dof(model=fa_sym, x0=x0_mpc, options=mpc_options)
    index_equi = 1  # index of equilibrium, since there are two with the simple arm
    # p_xy_ref = x_equis[index_equi, :]
    p_xyz_ref = np.array([0.0, 0.5, 0.0])
    controller.set_reference_cartesian(p_cartesian=p_xyz_ref, u=u_equi)

    # simulate
    ts = 0.01
    n_iter = 150
    sim = Simulator(fa, controller, 'RK45')
    x, u = sim.simulate(x0.flatten(), ts, 150)

    # Timing
    t_mean, t_std, t_min, t_max = controller.get_timing_statistics()
    print_timings(t_mean, t_std, t_min, t_max)

    # Plot result
    plot_result(x=x, u=u, t=np.arange(0, (n_iter + 1) * ts, ts))

    # Parse joint positions
    q = x[:, :fa.nq]

    # Animate simulated motion
    anim = Animator(fa, q, pos_ref=p_xy_ref).animate()
