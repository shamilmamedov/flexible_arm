#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import casadi as cs
import pinocchio as pin
from acados_template import AcadosModel


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
            self.K = np.diag([100.] * 4)
        else:
            assert (len(K) == 4)
            self.K = np.diag(K)

        # Process dampig parameters
        if D is None:
            self.D = np.diag([5.] * 4)
        else:
            assert (len(D) == 4)
            self.D = np.diag(D)

        path_to_urdf = 'models/flexible_arm_v1.urdf'

        # Try to load model from urdf file
        try:
            self.model = pin.buildModelFromUrdf(path_to_urdf)
        except ValueError:
            print(f"URDF file doesn't exist. Make sure path is correct!")

        # EE frame ID || 'load'
        self.ee_frame_id = self.model.getFrameId('link5')

        # Create data required for the algorithms
        self.data = self.model.createData()

        # Some useful parameters
        self.nx = self.model.nq + self.model.nv
        self.nq = self.model.nq
        self.nu = 1

    def fk(self, q, frame_id):
        """ Computes forward kinematics for a given frame

        :parameter q: a vector of joint configurations
        :parameter frame_id: an id of a desored frame
        """
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        # pin.updateFramePlacement(self.model, self.data, frame_id)
        T_EE_O = self.data.oMf[frame_id]
        R_EE_O = T_EE_O.rotation
        p_EE_O = T_EE_O.translation
        return R_EE_O, p_EE_O.reshape(-1,1)

    def fk_ee(self, q):
        """ Computes forward kinematics for EE frame in base frame
        """
        return self.fk(q, self.ee_frame_id)

    def frame_velocity(self, q, dq, frame_id):
        """ Computes end-effector velocity for a given frame

        :parameter q: joint configurations
        :parameter dq: joint velocities
        :parameter frame_id: an id of a desired frame
        """
        pin.forwardKinematics(self.model, self.data, q, dq)
        pin.updateFramePlacements(self.model, self.data)
        v = pin.getFrameVelocity(self.model, self.data,
                    frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

        return np.hstack((v.linear, v.angular)).reshape(-1,1)

    def ee_velocity(self, q, dq):
        """ Computes end-effector velocity
        """
        return self.frame_velocity(q, dq, self.ee_frame_id)

    def jacobian_ee(self, q):
        pass

    def gravity_torque(self, q):
        """ Computes gravity vector of the robot
        """
        t1 = np.zeros_like(q)
        return pin.rnea(self.model, self.data, q, t1, t1).reshape(-1, 1)

    def forward_dynamics(self, q, dq, tau):
        """ Computes forward dynamics of the robot
        """
        # Compute torque due to flexibility
        q_virt = q[1:,:]
        dq_virt = dq[1:,:]
        tau_flexibility = -self.K @ q_virt - self.D @ dq_virt
        tau_total = np.vstack((tau, tau_flexibility))
        return pin.aba(self.model, self.data, q, dq, tau_total).reshape(-1,1)

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
        return np.vstack((dq, self.forward_dynamics(q, dq, tau)))

    def fk_for_visualization(self, q):
        # Perform forward kinematics and get joint positions
        pin.forwardKinematics(self.model, self.data, q)
        p_joints = np.zeros((self.model.njoints + 1, 3))
        for k, oMi in enumerate(self.data.oMi):
            p_joints[k, :] = oMi.translation

        # Get also end-effector position
        pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
        p_joints[-1, :] = self.data.oMf[self.ee_frame_id].translation
        return p_joints

    def visualize(self, q):
        # Get joint positions
        p_joints = self.fk_for_visualization(q)

        # Plot the robot
        _, ax = plt.subplots()
        ax.plot(p_joints[:, 0], p_joints[:, 2], 'o-', lw=2, color='k')
        ax.scatter(p_joints[:-1, 0], p_joints[:-1, 2])
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_xlim([-0.55, 0.55])
        ax.set_ylim([-0.55, 0.55])
        plt.tight_layout()
        plt.show()


class SymbolicFlexibleArm:
    """ The class implements a model of the flexible arm in symbolic form
    using casadi. It is mainly intended to be used in MPC formulation.

    NOTE for time being majority of the parameters are fixed (HARDCODED)
    """

    def __init__(self, K=None, D=None) -> None:
        # Process stiffness parameters
        if K is None:
            self.K = np.diag([100.] * 4)
        else:
            assert (len(K) == 4)
            self.K = np.diag(K)

        # Process dampig parameters
        if D is None:
            self.D = np.diag([5.] * 4)
        else:
            assert (len(D) == 4)
            self.D = np.diag(D)

        # Number of joints, states and controls
        self.nq = 5
        self.nx = 2 * self.nq
        self.nu = 1
        self.nz = 2  # algebraic states
        self.length = 0.4  # todo Shamil: get length of model

        # constraints # todo Shamil: add constraints appropriately
        self.maximum_input_torque = 100  # Nm

        # Symbolic variables for joint positions, velocities and controls
        q = cs.MX.sym("q", self.nq)
        dq = cs.MX.sym("dq", self.nq)
        u = cs.MX.sym("u", self.nu)
        z = cs.MX.sym("z", self.nz)

        # Symbolic variables for dot values (needed for acados)
        x_dot = cs.MX.sym('xdot', self.nx)

        # Load the forward dynamics alogirthm function ABA
        casadi_aba = cs.Function.load('models/aba.casadi')

        # Compute torques of passive joints due to joint flexibility
        # Keep in mind only the first joint is active
        tau_p = -self.K @ q[1:] - self.D @ dq[1:]
        tau = cs.vertcat(u, tau_p)

        # Compute forward dynamics
        ddq = casadi_aba(q, dq, tau)

        # Compute right hand side of the system ODE
        rhs = cs.vertcat(dq, ddq)

        # compute positions
        self.rhs_impl = cs.vertcat(self.length * cs.cos(q[0]),
                                   self.length * cs.sin(q[0]))
        self.rhs_impl = cs.vertcat(self.length/4 * (cs.cos(q[0]) + cs.cos(q[0]+q[1])+ cs.cos(q[0]+q[1]+q[2])+ cs.cos(q[0]+q[1]+q[2]+q[3])),
                                   self.length/4 * (cs.sin(q[0]) + cs.sin(q[0]+q[1])+ cs.sin(q[0]+q[1]+q[2])+ cs.sin(q[0]+q[1]+q[2]+q[3])))
        self.x = cs.vertcat(q, dq)
        self.x_dot = x_dot
        self.u = u
        self.z = z
        self.rhs = rhs
        self.ode = cs.Function('ode', [self.x, self.u], [self.rhs],
                               ['x', 'u'], ['dx'])
        self.p_ee = cs.Function.load('models/fkp.casadi')
        self.v_ee = cs.Function.load('models/fkv.casadi')

    def get_acados_model(self) -> AcadosModel:
        model = AcadosModel()
        model.f_impl_expr = cs.vertcat(self.x_dot - self.rhs,
                                       self.z - self.rhs_impl)
        model.f_expl_expr = self.rhs
        model.x = self.x
        model.xdot = self.x_dot
        model.z = self.z
        model.u = self.u
        # model.p = w  # Not used right now
        model.name = "flexible_arm_nq" + str(self.nq)

        return model


if __name__ == "__main__":
    arm = FlexibleArm()
    print(arm.nq)

    sarm = SymbolicFlexibleArm()
    acados_model = sarm.get_acados_model()
    print(acados_model.name)
