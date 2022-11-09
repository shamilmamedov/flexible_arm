import numpy as np
import matplotlib.pyplot as plt

from flexible_arm_3dof import (SymbolicFlexibleArm3DOF, get_rest_configuration)
from ocp import OCPOptions, OptimalControlProblem
from animation import Panda3dAnimator
import plotting


def circle(c: np.ndarray, r: float, a: np.ndarray, 
           b: np.ndarray, t: np.ndarray):
    # a and b should be perpendicular
    assert np.dot(a.T, b).item() == 0

    out = c + r*np.cos(t)*a + r*np.sin(t)*b
    return out.T


def plot_circle(circ: np.ndarray, circ_ref: np.ndarray = None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D(circ[:,0], circ[:,1], circ[:,2])
    if circ_ref is not None:
        ax.scatter3D(circ_ref[:,0], circ_ref[:,1], circ_ref[:,2])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


def design_optimal_circular_trajectory(r: float = 0.2, tf: float = 0.7, visualize: bool = False):
    """
    :parameter r: radius of the circle
    :parameter tf: trajectory execution time
    """
    # Trajectory and model parameters/opts
    dt = 0.01
    N = int(tf/dt)

    # Specify the model
    n_seg_ocp = 0
    model = SymbolicFlexibleArm3DOF(n_seg=n_seg_ocp, dt=dt,
                                    integrator='cvodes')

    # Specify the initial rest position of the robot
    qa_t0 = np.array([0., 2*np.pi/5, -np.pi/3])
    q_t0 = get_rest_configuration(qa_t0, n_seg_ocp)
    pee_t0 = np.array(model.p_ee(q_t0))

    # Design circle reference for the end-effector
    c = pee_t0 + np.array([[0., 0., -r]]).T
    a = np.array([[0., 0., 1.]]).T
    b = np.array([[0., 1., 0.]]).T
    s = np.linspace(0., 2*np.pi, N+1) 
    pee_ref = circle(c, r, a, b, s)

    # Design and solve OCP
    opts = OCPOptions(dt, tf)
    assert opts.N == N

    ocp = OptimalControlProblem(model, pee_ref, opts)
    t_opt, q_opt, dq_opt, u_opt = ocp.solve(q_t0)


    # Compute forward kinematics
    N = opts.N
    pee_opt = np.zeros((N+1, 3))
    for k in range(N+1):
        pee_opt[k,:] = np.array(model.p_ee(q_opt[k,:])).flatten()

    if visualize:
        # Plot optimnal trajectories
        plotting.plot_ee_positions(t_opt, pee_opt, pee_ref)
        plot_circle(pee_ref, pee_opt)
        # plotting.plot_joint_positions(t_opt, q_opt, n_seg_ocp)
        plotting.plot_joint_velocities(t_opt, dq_opt, n_seg_ocp)
        plotting.plot_controls(t_opt[:-1], u_opt)

        # Visualize optimal motion
        Panda3dAnimator(model.urdf_path, dt, q_opt).play(5)

    return t_opt, q_opt, dq_opt, u_opt, pee_opt


if __name__ == "__main__":
    design_optimal_circular_trajectory(visualize=True)