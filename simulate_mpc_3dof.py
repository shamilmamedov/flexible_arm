#!/usr/bin/env python3
from copy import deepcopy
import numpy as np
import pinocchio as pin
from utils import plot_result, print_timings, ControlMode
from flexible_arm_3dof import FlexibleArm3DOF, SymbolicFlexibleArm3DOF
from animation import Panda3dAnimator
from mpc_3dof import Mpc3dofOptions, Mpc3Dof
from simulation import Simulator

if __name__ == "__main__":
    # options
    control_mode = ControlMode.REFERENCE_TRACKING

    # Create FlexibleArm instance
    n_seg = 3
    n_seg_mpc = n_seg
    fa = FlexibleArm3DOF(n_seg)
    fa_sym = SymbolicFlexibleArm3DOF(n_seg_mpc)

    # Sample a random configuration
    q = pin.randomConfiguration(fa.model)
    # fa.visualize(q)

    # Create initial state simulated system
    q0 = np.zeros((fa.nq, 1))
    q0[0] += 0.5
    q0[1] += 1.5
    q0[1 + n_seg + 1] += 0.5

    dq0 = np.zeros_like(q0)
    x0 = np.vstack((q0, dq0))

    # MPC has other model states
    q_mpc0 = deepcopy(q0)
    dq_mpc0 = deepcopy(dq0)
    x_mpc0 = np.vstack((q_mpc0, dq_mpc0))

    _, qee0 = fa.fk_ee(q)

    # Compute reference
    q_mpc_ref = deepcopy(q_mpc0)
    q_mpc_ref[0] += 1.5
    q_mpc_ref[1] += 0.5
    q_mpc_ref[1 + n_seg + 1] += 1.5

    dq_mpc_ref = np.zeros_like(q_mpc_ref)
    x_mpc_ref = np.vstack((q_mpc_ref, dq_mpc_ref))
    _, x_ee_ref = fa.fk_ee(q_mpc_ref)

    # Create mpc options and controller
    mpc_options = Mpc3dofOptions(n_links=n_seg_mpc)
    controller = Mpc3Dof(model=fa_sym, x0=x_mpc0, x0_ee=qee0, options=mpc_options)
    u_ref = np.zeros((fa_sym.nu, 1))  # u_ref could be changed to some known value along the trajectory

    # Choose one out of two control modes. Reference tracking uses a spline planner.
    if control_mode == ControlMode.SET_POINT:
        controller.set_reference_point(p_ee_ref=x_ee_ref, x_ref=x_mpc_ref, u_ref=u_ref)
    elif control_mode == ControlMode.REFERENCE_TRACKING:
        controller.set_reference_trajectory(q_t0=q_mpc0, pee_tf=x_ee_ref, tf=5, fun_forward_pee=fa.fk_ee)
    else:
        Exception()

    # simulate
    ts = 0.01
    n_iter = 600
    sim = Simulator(fa, controller, 'RK45')
    x, u = sim.simulate(x0.flatten(), ts, n_iter)

    # Print timing
    t_mean, t_std, t_min, t_max = controller.get_timing_statistics()
    print_timings(t_mean, t_std, t_min, t_max)

    # Plot result
    # plot_result(x=x, u=u, t=np.arange(0, (n_iter + 1) * ts, ts))

    # Parse joint positions
    q = x[::10, :fa.nq]

    # Animate simulated motion
    urdf_path = 'models/three_dof/three_segments/flexible_arm_3dof_3s.urdf'
    animator = Panda3dAnimator(urdf_path, 0.01, q).play(3)
