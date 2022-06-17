#!/usr/bin/env python3

import numpy as np
import pinocchio as pin
from utils import plot_result, print_timings
from flexible_arm import FlexibleArm, SymbolicFlexibleArm
from animation import Animator
from mpc import MpcOptions, Mpc
from simulation import simulate_closed_loop
from controller import ConstantController

if __name__ == "__main__":
    # Create FlexibleArm instance
    fa = FlexibleArm(K=[7.] * 4)
    fa_sym = SymbolicFlexibleArm(K=[7.] * 4)

    # Sample a random configuration
    q = pin.randomConfiguration(fa.model)
    # fa.visualize(q)

    # Simulate
    q = np.zeros((fa.nq, 1))
    dq = np.zeros_like(q)
    x0 = np.vstack((q, dq))

    mpc_options = MpcOptions()
    controller = Mpc(model=fa_sym, x0=x0, options=mpc_options)

    ts = 0.01
    n_iter = 150
    x, u = simulate_closed_loop(ts, n_iter, fa, controller, x0.flatten())

    # Timing
    t_mean, t_std, t_min, t_max = controller.get_timing_statistics()
    print_timings(t_mean, t_std, t_min, t_max)

    # Plot result
    plot_result(x=x, u=u, t=np.arange(0, (n_iter+1)*ts, ts))

    # Parse joint positions
    q = x[:, :fa.nq]

    # Animate simulated motion
    anim = Animator(fa, q).animate()
