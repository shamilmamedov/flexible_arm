#!/usr/bin/env python3

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import casadi as cs
import pinocchio as pin
from acados_template import AcadosModel


class FlexibleArm3DOF:
    """ Implements flexible arm robot by dividing a link into
    a several smaller virtual links with virtual joint in between. Joints
    are passive, but have stiffness and damping.

    First joint is an active joint i.e. it can controlled by commading torque

    NOTE for now the number of virtual joints and links are fixed
    """

    def __init__(self, n_seg) -> None:
        """ Class constructor. Based on the number of segments, it loads
        a urdf file and flexbility parameters defined in a yaml file

        :parameter n_seg: number of segments for the flexible link

        TODO:
            1. Extend the model with flexibility in the first joint (SEA like model)
        """
        # Sanity checks
        assert (n_seg in [3, 5, 10])

        # Build urdf path
        n_seg_int2str = {1: 'one', 3: 'three', 5: 'five', 10: 'ten'}

        model_folder = 'models/three_dof/' + n_seg_int2str[n_seg] + '_segments/'
        urdf_file = 'flexible_arm_3dof_' + str(n_seg) + 's.urdf'
        urdf_path = os.path.join(model_folder, urdf_file)

        # Try to load model from urdf file
        try:
            self.model = pin.buildModelFromUrdf(urdf_path)
        except ValueError:
            print(f"URDF file doesn't exist. Make sure path is correct!")

        # EE frame ID || 'load'
        ee_frame_name = 'load'
        if self.model.existFrame(ee_frame_name):
            self.ee_frame_id = self.model.getFrameId(ee_frame_name)
        else:
            raise ValueError

        # Create data required for the algorithms
        self.data = self.model.createData()

        # Useful model parameters
        self.n_seg = n_seg
        self.nx = self.model.nq + self.model.nv
        self.nq = self.model.nq
        self.nu = 3

        # Process flexibility parameters
        params_file = 'flexibility_params.yml'
        params_path = os.path.join(model_folder, params_file)

        with open(params_path) as f:
            flexibility_params = yaml.safe_load(f)

        # Additional sanity checks
        self.K2 = np.diag(flexibility_params['K2'][1:])
        self.K3 = np.diag(flexibility_params['K3'][1:])
        self.D2 = np.diag(flexibility_params['D2'][1:])
        self.D3 = np.diag(flexibility_params['D3'][1:])

    def random_q(self):
        """ Returns a random configuration
        """
        return pin.randomConfiguration(self.model)

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
        return R_EE_O, p_EE_O.reshape(-1, 1)

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

        return np.hstack((v.linear, v.angular)).reshape(-1, 1)

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
        if len(tau.shape) < 2:
            tau = np.expand_dims(tau, 1)
        # Compute torque due to flexibility
        qp_link2 = q[2:2 + self.n_seg]
        qp_link3 = q[2 + self.n_seg + 1:]
        dqp_link2 = dq[2:2 + self.n_seg]
        dqp_link3 = dq[2 + self.n_seg + 1:]
        taup_link2 = -self.K2 @ qp_link2 - self.D2 @ dqp_link2
        taup_link3 = -self.K3 @ qp_link3 - self.D3 @ dqp_link3
        tau_total = np.vstack((tau[:2, :], taup_link2, tau[2, :], taup_link3))

        return pin.aba(self.model, self.data, q, dq, tau_total).reshape(-1, 1)

    def inverse_dynamics(self, q, dq, ddq):
        """ Computes inverse dynamics of the robot
        """
        return pin.rnea(self.model, self.data, q, dq, ddq)

    def ode(self, x, tau):
        """ Computes ode of the robot
        :parameter x: [10x1] vector of robot states
        :parameter tau: [1x1] vector of an active joint torque
        """
        q = x[0:self.nq, :]
        dq = x[self.nq:, :]
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


class SymbolicFlexibleArm3DOF:
    """ The class implements a model of a 3dof flexible arm in symbolic form
    using casadi. It is mainly intended to be used in MPC formulation.
    First link is rigid!
    
    Class attributes
        x - a vector of symbolic variables for states
        u - a vector of symbolic variables for controls
        rhs - a symbolic expression for the right-hand side of ODE
        ode - a casadi function for evaluating rhs
        p_ee - a casadi function for evaluating forward kinematics (ee position)
        v_ee - a casadi function for evaluating ee velocity

    """

    def __init__(self, n_seg) -> None:
        """ Class constructor
        """
        # Sanity checks
        assert (n_seg in [3, 5, 10])

        # Path to a folder with model description
        n_seg_int2str = {1: 'one', 3: 'three', 5: 'five', 10: 'ten'}
        model_folder = 'models/three_dof/' + n_seg_int2str[n_seg] + '_segments/'

        # Number of joints, states and controls
        self.nq = 1 + 2 * (n_seg + 1)
        self.nx = 2 * self.nq
        self.nu = 3
        self.nz = 3  # algebraic states

        # Process flexibility parameters
        params_file = 'flexibility_params.yml'
        params_path = os.path.join(model_folder, params_file)

        with open(params_path) as f:
            flexibility_params = yaml.safe_load(f)

        self.K2 = np.diag(flexibility_params['K2'][1:])
        self.K3 = np.diag(flexibility_params['K3'][1:])
        self.D2 = np.diag(flexibility_params['D2'][1:])
        self.D3 = np.diag(flexibility_params['D3'][1:])

        # Symbolic variables for joint positions, velocities and controls
        q = cs.MX.sym("q", self.nq)
        dq = cs.MX.sym("dq", self.nq)
        u = cs.MX.sym("u", self.nu)
        z = cs.MX.sym("z", self.nz)

        # Load the forward dynamics alogirthm function ABA
        casadi_aba = cs.Function.load(os.path.join(model_folder, 'aba.casadi'))

        # Compute torques of passive joints due to joint flexibility
        # Keep in mind only the first joint is active
        qp_link2 = q[2:2 + n_seg]
        qp_link3 = q[2 + n_seg + 1:]
        dqp_link2 = dq[2:2 + n_seg]
        dqp_link3 = dq[2 + n_seg + 1:]
        taup_link2 = -self.K2 @ qp_link2 - self.D2 @ dqp_link2
        taup_link3 = -self.K3 @ qp_link3 - self.D3 @ dqp_link3
        tau = cs.vertcat(u[:2], taup_link2, u[2], taup_link3)

        # Compute forward dynamics
        ddq = casadi_aba(q, dq, tau)

        # Compute right hand side of the system ODE
        rhs = cs.vertcat(dq, ddq)

        self.x = cs.vertcat(q, dq)
        self.u = u
        self.rhs = rhs
        self.ode = cs.Function('ode', [self.x, self.u], [self.rhs],
                               ['x', 'u'], ['dx'])
        self.p_ee = cs.Function.load(os.path.join(model_folder, 'fkp.casadi'))
        self.v_ee = cs.Function.load(os.path.join(model_folder, 'fkv.casadi'))

        # Symbolic variables for dot values (needed for acados)
        x_dot = cs.MX.sym('xdot', self.nx)

        self.rhs_impl = self.p_ee(q)
        self.x = cs.vertcat(q, dq)
        self.x_dot = x_dot
        self.u = u
        self.z = z
        self.rhs = rhs
        self.ode = cs.Function('ode', [self.x, self.u], [self.rhs],
                               ['x', 'u'], ['dx'])

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
    # n_seg = 3
    # arm = FlexibleArm(n_seg)
    # q = np.zeros((arm.nq, 1))
    # R, p = arm.fk_ee(q)
    # print(R, p)

    # # arm.visualize(q)

    # sarm = SymbolicFlexibleArm(n_seg)
    # print(sarm.p_ee(q))
    sarm = SymbolicFlexibleArm3DOF(n_seg=10)
