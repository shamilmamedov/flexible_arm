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
    n_seg = 3
    n_seg_mpc = n_seg
    fa = FlexibleArm3DOF(n_seg)
    fa_sym = SymbolicFlexibleArm3DOF(n_seg_mpc)

    # Sample a random configuration
    q = pin.randomConfiguration(fa.model)
    # fa.visualize(q)

    # Simulate
    q = np.zeros((fa.nq, 1))
    dq = np.zeros_like(q)
    x0 = np.vstack((q, dq))

    # MPC has other model states
    q_mpc = np.zeros((fa_sym.nq, 1))
    dq_mpc = np.zeros_like(q_mpc)
    x0_mpc = np.vstack((q_mpc, dq_mpc))

    _, x0_ee = fa.fk_ee(q)

    # Compute reference
    q_ref = np.zeros((fa.nq, 1))
    q_ref[0] += 1.
    q_ref[n_seg+1] += 1
    dq_ref = np.zeros_like(q)
    x_ref = np.vstack((q_ref, dq_ref))
    _, x_ee_ref = fa.fk_ee(q_ref)

    mpc_options = Mpc3dofOptions(n_links=n_seg_mpc)
    controller = Mpc3Dof(model=fa_sym, x0=x0_mpc, x0_ee=x0_ee, options=mpc_options)
    u_ref = np.zeros((fa_sym.nu, 1))
    controller.set_reference_cartesian(p_ee_ref=x_ee_ref, x_ref=x_ref, u_ref=u_ref)

    # simulate
    ts = 0.01
    n_iter = 251
    sim = Simulator(fa, controller, 'RK45')
    x, u = sim.simulate(x0.flatten(), ts, n_iter)

    # Timing
    t_mean, t_std, t_min, t_max = controller.get_timing_statistics()
    print_timings(t_mean, t_std, t_min, t_max)

    # Plot result
    plot_result(x=x, u=u, t=np.arange(0, (n_iter + 1) * ts, ts))

    # Parse joint positions
    q = x[:, :fa.nq]

    # Animate simulated motion
    # anim = Animator(fa, q, pos_ref=p_xy_ref).animate()
