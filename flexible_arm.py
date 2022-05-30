#!/usr/bin/env python3

import numpy as np
import pinocchio as pin
from numpy import pi
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from animation import Animator
from controller import DummyController, PDController


class FlexibleArm:
    """ Implements flexible arm robot by dividing a link into
    a several smaller virtual links with virtual joint in between. Joints
    are passive, but have stiffness and damping.

    First joint is an active joint i.e. it can controlled by commading torque

    NOTE for now the number of virtual joints and links are fixed
    """
    def __init__(self, K=None, D=None) -> None:
        """ Class constructor
        :parameter K: an array of stiffnesses of passive virtual joints
        :parameter D: an array of dampings of passive virtual joints
        """
        # Process stiffness parameters
        if K is None:
            self.K = np.diag([100.]*4)
        else:
            assert(K.size==4)
            self.K = np.diag(K)

        # Process dampig parameters
        if D is None:
            self.D = np.diag([5.]*4)
        else:
            assert(D.size==4)
            self.D = np.diag(D)

        path_to_urdf = 'flexible_arm_v1.urdf'
        
        # Try to load model from urdf file
        try:
            self.model = pin.buildModelFromUrdf(path_to_urdf)
        except ValueError:
            print(f"URDF file doesn't exist. Make sure path is correct!")

        # EE frame ID || 'load'
        self.ee_frame_id = self.model.getFrameId('link5_to_load') 

        # Create data required for the algorithms
        self.data = self.model.createData()

        # Some useful parameters
        self.nx = self.model.nq + self.model.nv
        self.nq = self.model.nq
        self.nu = 1

    def fk_ee(self, q):
        pass

    def jacobian_ee(self, q):
        pass

    def gravity_torque(self, q):
        """ Computes gravity vector of the robot
        """
        t1 = np.zeros_like(q)
        return pin.rnea(self.model, self.data, q, t1, t1).reshape(-1,1)

    def forward_dynamics(self, q, dq, tau):
        """ Computes forward dynamics of the robot
        """
        return pin.aba(self.model, self.data, q, dq, tau)

    def inverse_dynamics(self, q, dq, ddq):
        """ Computes inverse dynamics of the robot
        """
        return pin.rnea(self.model, self.data, q, dq, ddq)

    def ode(self, x, tau):
        """ Computes ode of the robot
        :parameter x: [10x1] vector of robot states
        :parameter tau: [1x1] vector of an active joint torque
        """
        q = x[0:self.nq,:]
        dq = x[self.nq:,:]
        q_virt = q[1:,:]
        dq_virt = dq[1:,:]
        tau_flexibility = -self.K @ q_virt - self.D @ dq_virt
        tau_total = np.vstack((tau, tau_flexibility))
        return np.vstack((dq, pin.aba(self.model, self.data, q, dq, tau_total).reshape(-1,1)))

    def fk_for_visualization(self, q):
        # Perform forward kinematics and get joint positions
        pin.forwardKinematics(self.model, self.data, q)
        p_joints = np.zeros((self.model.njoints+1,3))
        for k, oMi in enumerate(self.data.oMi):
            p_joints[k,:] = oMi.translation
        
        # Get also end-effector position
        pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
        p_joints[-1,:] = self.data.oMf[self.ee_frame_id].translation
        return p_joints

    def visualize(self, q):
        # Get joint positions
        p_joints = self.fk_for_visualization(q)

        # Plot the robot
        _, ax = plt.subplots()
        ax.plot(p_joints[:,0], p_joints[:,2], 'o-', lw=2, color='k')
        ax.scatter(p_joints[:-1,0], p_joints[:-1,2])
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_xlim([-0.55, 0.55])
        ax.set_ylim([-0.55, 0.55])
        plt.tight_layout()
        plt.show()


def robot_ode(t, x, robot, tau):
    """ Wrapper on ode of the robot class
    """
    return robot.ode(x, tau)

def simulate_closed_loop(ts, n_iter, robot, controller, x0):
    x = np.zeros((n_iter+1, robot.nx))
    u = np.zeros((n_iter, robot.nu))
    x[0,:] = x0
    for k in range(n_iter):
        qk = x[[k],:robot.nq].T
        dqk = x[[k],robot.nq:].T
        
        tau = controller.compute_torques(qk[0], dqk[0])
        u[[k],:] = tau

        sol = solve_ivp(robot_ode, [0, ts], x[k,:], args=(robot, tau), vectorized=True)
        x[k+1,:] = sol.y[:,-1]

    return x, u


if __name__ == "__main__":
    # Create FlexibleArm instance
    fa = FlexibleArm()

    # Sample a random configuration
    q = pin.randomConfiguration(fa.model)
    # fa.visualize(q)

    # Simulate
    q = np.zeros((fa.nq, 1))
    dq = np.zeros_like(q)
    x0 = np.vstack((q, dq))

    # controller = DummyController()
    controller = PDController(Kp=150, Kd=5, q_ref=np.array([np.pi/4]))

    ts = 0.01
    n_iter = 150
    x, u = simulate_closed_loop(ts, n_iter, fa, controller, x0.flatten())    

    # Parse joint positions
    q = x[:,:fa.nq]

    # Animate simulated motion
    anim = Animator(fa, q).animate()