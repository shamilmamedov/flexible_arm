#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from flexible_arm import FlexibleArm 
from controller import PDController
from simulation import simulate_closed_loop
from animation import Animator


def compare_jiminy_vs_own():
    # Load solution from Jiminy
    # tqu contains time, joint positions and torque
    tq_jiminy = np.loadtxt('data/tqu.csv', delimiter=',')
    t_jiminy = tq_jiminy[:,0]
    q_jiminy = tq_jiminy[:,1:-1]
    tau_jiminy = tq_jiminy[:,-1]

    # Instantiate a robot
    K = np.array([100.]*4)
    D = np.array([5.]*4)
    fa = FlexibleArm(K, D)

    # Instantiate a controller
    controller = PDController(Kp=150., Kd=5., q_ref=np.array([np.pi/4]))

    # Set initial states
    q = np.zeros((fa.nq, 1))
    dq = np.zeros_like(q)
    x0 = np.vstack((q, dq))

    # Set simulator parameters
    ts = 0.001
    n_iter = 2000

    # Simulate
    x, u = simulate_closed_loop(ts, n_iter, fa, controller, x0.flatten()) 

    # Parse joint positions
    t = np.arange(0, n_iter+1)*ts
    q = x[:,:fa.nq]

    _, ax_qa = plt.subplots()
    ax_qa.plot(t, q[:,0])
    ax_qa.plot(t_jiminy, q_jiminy[:,0], '--')
    # plt.show()

    _, ax_qf = plt.subplots()
    for k, (qf_me, qf_jiminy) in enumerate(zip(q[:,1:].T, q_jiminy[:,1:].T)):
        ax_qf.plot(t, qf_me, label=f'qf {k+1}')
        ax_qf.plot(t_jiminy, qf_jiminy, '--')
    ax_qf.legend()

    _, ax_tau = plt.subplots()
    ax_tau.plot(t[:-1], u)
    ax_tau.plot(t_jiminy, tau_jiminy, '--')
    plt.show()

def animate_jiminy_solution():
    # Load solution from Jiminy
    tq_jiminy = np.loadtxt('data/tqu.csv', delimiter=',')
    q_jiminy = tq_jiminy[:,1:-1]

    # Instantiate a robot
    fa = FlexibleArm()

    # Animate simulated motion
    anim = Animator(fa, q_jiminy[::10,:]).animate()


if __name__ == "__main__":
    compare_jiminy_vs_own()
    # animate_jiminy_solution()