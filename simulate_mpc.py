#!/usr/bin/env python3

import numpy as np
import pinocchio as pin
from scipy.integrate import solve_ivp
from flexible_arm import FlexibleArm, SymbolicFlexibleArm
from animation import Animator
from mpc import MpcOptions, Mpc
from simulation import simulate_closed_loop

if __name__ == "__main__":
    # Create FlexibleArm instance
    fa = FlexibleArm()
    fa_sym = SymbolicFlexibleArm()

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

    # Parse joint positions
    q = x[:, :fa.nq]

    # Animate simulated motion
    anim = Animator(fa, q).animate()
