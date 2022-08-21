#!/usr/bin/env python3

import numpy as np
import pinocchio as pin
from compute_equilibria import EquilibriaWrapper
from utils import plot_result, print_timings
from flexible_arm import FlexibleArm, SymbolicFlexibleArm
from animation import Animator
from mpc import MpcOptions, Mpc
from simulation import Simulator
from controller import ConstantController, PDController

if __name__ == "__main__":
    # Create FlexibleArm instance
    # Create FlexibleArm instance
    n_seg = 3
    n_seg_mpc = 3
    fa = FlexibleArm(n_seg)
    fa_sym = SymbolicFlexibleArm(n_seg_mpc)

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

    #equi_wrapper = EquilibriaWrapper(model_sym=fa_sym, model=fa)
    u_equi = -7
    #x_equis = equi_wrapper.get_equilibrium_cartesian_states(input_u=u_equi)
    #equi_wrapper.plot()

    mpc_options = MpcOptions(n_links=n_seg_mpc)
    controller = Mpc(model=fa_sym, x0=x0_mpc, options=mpc_options)
    index_equi = 0  # index of equilibrium, since there are two with the simple arm
    #p_xy_ref = x_equis[index_equi, :]
    p_xy_ref=np.array([0.5,0.1])
    controller.set_reference_cartesian(x=p_xy_ref[0], y=p_xy_ref[1], u=u_equi)

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
