#!/usr/bin/env python3

import matplotlib
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from flexible_arm_3dof import FlexibleArm3DOF, SymbolicFlexibleArm3DOF
from animation import Animator, Panda3dAnimator
from controller import DummyController, PDController3Dof
from simulation import Simulator

if __name__ == "__main__":
    # Create FlexibleArm instance
    n_seg = 3
    fa = FlexibleArm3DOF(n_seg)

    # Sample a random configuration
    q = pin.randomConfiguration(fa.model)
    # fa.visualize(q)

    # Simulate
    q0 = np.zeros((fa.nq, 1))
    q0[0] += 0.5
    q0[1] += 1.5
    q0[1 + n_seg + 1] += 0.5

    dq0 = np.zeros_like(q0)
    x0 = np.vstack((q0, dq0))

    # Compute reference
    q_ref = np.zeros((fa.nq, 1))
    q_ref[0] += 1.5
    q_ref[1] += 0.5
    q_ref[1 + n_seg + 1] += 1.5

    # controller = DummyController()
    controller = PDController3Dof(Kp=(40, 40, 40), Kd=(0.25, 0.25, 0.25),
                                  n_seg=n_seg, q_ref=q_ref)

    ts = 0.001
    n_iter = 5000

    sim = Simulator(fa, controller, 'LSODA')
    x, u = sim.simulate(x0.flatten(), ts, n_iter)
    t = np.arange(0, n_iter + 1) * ts

    # Parse joint positions
    q = x[::10, :fa.nq]

    _, ax = plt.subplots()
    ax.plot(t[::10], q[:, 0])
    # plt.show()

    # Animate simulated motion
    # anim = Animator(fa, q).play()

    urdf_path = 'models/three_dof/three_segments/flexible_arm_3dof_3s.urdf'
    animator = Panda3dAnimator(urdf_path, 0.01, q).play(3)
