import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import casadi as cs
import pinocchio as pin
from scipy.linalg import block_diag
from acados_template import AcadosModel

from integrator import symbolic_RK4
from animation import Panda3dAnimator
from copy import deepcopy

n_seg_int2str = {0: 'zero', 1:'one', 2:'two', 3: 'three', 5: 'five', 10: 'ten'}


class FlexibleArm3DOF:
    """ Implements flexible arm robot by dividing a link into
    a several smaller virtual links with virtual joint in between. Joints
    are passive, but have stiffness and damping.

    First joint is an active joint i.e. it can be controlled by commanding torque

    NOTE for now the number of virtual joints and links are fixed
    """

    def __init__(self, n_seg) -> None:
        """ Class constructor. Based on the number of segments, it loads
        an urdf-file and flexibility parameters defined in a yaml file

        :parameter n_seg: number of segments for the flexible link
        """
        # Sanity checks
        assert (n_seg in [0, 1, 2, 3, 5, 10])

        # Build urdf path
        model_folder = 'models/three_dof/' + \
                       n_seg_int2str[n_seg] + '_segments/'
        urdf_file = 'flexible_arm_3dof_' + str(n_seg) + 's.urdf'
        self.urdf_path = os.path.join(model_folder, urdf_file)

        # Try to load model from urdf file
        try:
            self.model = pin.buildModelFromUrdf(self.urdf_path)
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
        self.ny = 9
        self.qa_idx = [0, 1, 2 + n_seg] # indexes of the active joints

        # Process flexibility parameters
        if self.n_seg > 0:
            params_file = 'flexibility_params.yml'
            params_path = os.path.join(model_folder, params_file)

            with open(params_path) as f:
                flexibility_params = yaml.safe_load(f)

            # Additional sanity checks
            self.K2 = np.diag(flexibility_params['K2'])
            self.K3 = np.diag(flexibility_params['K3'])
            self.D2 = np.diag(flexibility_params['D2'])
            self.D3 = np.diag(flexibility_params['D3'])

    def random_q(self):
        """ Returns a random configuration for the active joints
        and zero position for the passive joints
        """
        q = np.zeros(self.nq)
        q[self.qa_idx] = pin.randomConfiguration(self.model)[self.qa_idx]
        return q

    def random_qa(self):
        """ Returns a random configuration for active joints 
        """
        return pin.randomConfiguration(self.model)[self.qa_idx]

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
        return deepcopy(self.fk(q, self.ee_frame_id))

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
        if tau.shape[0] < tau.shape[1]:
            tau = tau.transpose()

        # Compute torque due to flexibility
        if self.n_seg > 0:
            qp_link2 = q[2:2 + self.n_seg]
            qp_link3 = q[2 + self.n_seg + 1:]
            dqp_link2 = dq[2:2 + self.n_seg]
            dqp_link3 = dq[2 + self.n_seg + 1:]
            taup_link2 = -self.K2 @ qp_link2 - self.D2 @ dqp_link2
            taup_link3 = -self.K3 @ qp_link3 - self.D3 @ dqp_link3
            tau_total = np.vstack(
                (tau[:2, :], taup_link2, tau[2, :], taup_link3))
        else:
            tau_total = tau

        return pin.aba(self.model, self.data, q, dq, tau_total).reshape(-1, 1)

    def inverse_dynamics(self, q, dq, ddq):
        """ Computes inverse dynamics of the robot
        """
        return pin.rnea(self.model, self.data, q, dq, ddq)

    def ode(self, x, tau):
        """ Computes ode of the robot
        :parameter x: [nx x 1] vector of robot states
        :parameter tau: [nu x 1] vector of an active joint torque
        """
        x = x.reshape(-1, 1)
        q = x[0:self.nq, :]
        dq = x[self.nq:, :]
        return np.vstack((dq, self.forward_dynamics(q, dq, tau)))

    def output(self, x):
        """
        :parameter x: [nx x 1] vector of robot states
        """
        q = x[0:self.nq, :]
        dq = x[self.nq:, :]
        qa = q[[0, 1, 2 + self.n_seg], :]
        dqa = dq[[0, 1, 2 + self.n_seg], :]
        pee = self.fk_ee(q)[1]
        return np.vstack((qa, dqa, pee))


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
        y - a symbolic expression for the output, it includes active
            joint positions, active joint velocities and the end-effector position
        F - a symbolic integrator of the dynamics equation x(k+1) = F(x(k), u(k)) 
            For now the integrator is 1 step RK4 integrator and sampling time is a free
            variable that should be provided during numerial integration.
    """

    def __init__(self, n_seg: int, integrator: str = 'collocation', dt: float = 0.001) -> None:
        """ Class constructor

        :parameter n_seg: number of segments of a link
        :parameter integrator: an integration scheme used to get discrete map
        :parameter dt: integration step
        """
        # Sanity checks
        assert (n_seg in [0, 1, 2, 3, 5, 10])
        assert (integrator in ['cvodes', 'idas', 'collocation'])

        # Path to a folder with model description
        model_folder = 'models/three_dof/' + \
                       n_seg_int2str[n_seg] + '_segments/'
        urdf_file = 'flexible_arm_3dof_' + str(n_seg) + 's.urdf'
        self.urdf_path = os.path.join(model_folder, urdf_file)

        # Dimensionality parameters
        self.nq = 1 + 2 * (n_seg + 1)
        self.nx = 2 * self.nq
        self.nu = 3
        self.nz = 3  # algebraic states
        self.ny = 9  # 3 for active joint positions
        self.n_seg = n_seg
        self.qa_idx = [0, 1, 2 + n_seg] # indeces of the active joitns

        # Velocity and torque limits
        self.tau_max = np.array([20, 10, 10])
        self.dqa_max = np.array([2.5, 3.5, 3.5])

        # Process flexibility parameters
        if self.n_seg > 0:
            params_file = 'flexibility_params.yml'
            params_path = os.path.join(model_folder, params_file)

            with open(params_path) as f:
                flexibility_params = yaml.safe_load(f)

            self.K2 = np.diag(flexibility_params['K2'])
            self.K3 = np.diag(flexibility_params['K3'])
            self.D2 = np.diag(flexibility_params['D2'])
            self.D3 = np.diag(flexibility_params['D3'])

        # Load the forward dynamics alogirthm function ABA and RNEA
        casadi_aba = cs.Function.load(os.path.join(model_folder, 'aba.casadi'))
        self.rnea = cs.Function.load(os.path.join(model_folder, 'rnea.casadi'))

        # Get function for the forward kinematics and velocities
        self.p_ee = cs.Function.load(os.path.join(model_folder, 'fkp.casadi'))
        self.v_ee = cs.Function.load(os.path.join(model_folder, 'fkv.casadi'))

        # Symbolic variables for joint positions, velocities and controls
        q = cs.MX.sym("q", self.nq)
        dq = cs.MX.sym("dq", self.nq)
        u = cs.MX.sym("u", self.nu)
        z = cs.MX.sym("z", self.nz)
        x = cs.vertcat(q, dq)
        x_dot = cs.MX.sym('xdot', self.nx) # needed for acados

        if self.n_seg > 0:
            # Compute torques of passive joints due to joint flexibility
            # Keep in mind only the first joint is active
            qa = q[[0, 1, 2 + n_seg]]
            qp_link2 = q[2:2 + n_seg]
            qp_link3 = q[2 + n_seg + 1:]

            dqp_link2 = dq[2:2 + n_seg]
            dqp_link3 = dq[2 + n_seg + 1:]
            dqa = dq[[0, 1, 2 + n_seg]]

            taup_link2 = -self.K2 @ qp_link2 - self.D2 @ dqp_link2
            taup_link3 = -self.K3 @ qp_link3 - self.D3 @ dqp_link3
            tau = cs.vertcat(u[:2], taup_link2, u[2], taup_link3)
        else:
            qa = q
            dqa = dq
            tau = u      

        # Accelerations and right-hand-side of the ODE
        ddq = casadi_aba(q, dq, tau)
        rhs = cs.vertcat(dq, ddq)
        h = cs.vertcat(qa, dqa, self.p_ee(q))

        # Calculate derivatives of the rhs and output wrt control and states to get linearized
        # dynamics to be used in state observers for example
        drhs_dx = cs.jacobian(rhs, x)
        drhs_du = cs.jacobian(rhs, u)
        dh_dx = cs.jacobian(h, x)

        self.rhs_impl = self.p_ee(q)
        self.x = x
        self.x_dot = x_dot
        self.u = u
        self.z = z
        self.rhs = rhs
        self.ode = cs.Function('ode', [self.x, self.u], [self.rhs],
                               ['x', 'u'], ['dx'])
        self.h = cs.Function('h', [self.x], [h], ['x'], ['h'])

        self.df_dx = cs.Function('df_dx', [self.x, self.u], [drhs_dx], 
                                 ['x', 'u'], ['df_dx'])
        self.df_du = cs.Function('df_du', [self.x, self.u], [drhs_du], 
                                 ['x', 'u'], ['df_du'])
        self.dh_dx = cs.Function('dy_dx', [self.x], [dh_dx], 
                                 ['x'], ['dy_dx'])


        # Create an integrator
        dae = {'x': self.x, 'p': self.u, 'ode': self.rhs}
        if integrator == 'collocation':
            opts = {'t0': 0, 'tf': dt, 'number_of_finite_elements': 3, 
                    'simplify': True, 'collocation_scheme': 'radau',
                    'rootfinder':'fast_newton','expand': True,  # fast_newton, newton
                    'interpolation_order': 3}
        elif integrator == 'cvodes':
            opts = {'t0': 0, 'tf': dt, 
                    'nonlinear_solver_iteration': 'functional', # 'expand': True,
                    'linear_multistep_method': 'bdf'}
        I = cs.integrator('I', integrator, dae, opts)
        x_next = I(x0=self.x, p=self.u)["xf"]
        self.F = cs.Function('F', [x, u], [x_next])
        self.dF_dx = cs.Function('dF_dx', [x, u], [cs.jacobian(x_next, x)])

    def get_acados_model(self) -> AcadosModel:
        """ Return an acados model later used in OCP
        Implicit dynamics is extended with an algebraic state that
        corresponds to the end-effector position (I guess it is needed
        for tracking)
        """
        model = AcadosModel()
        model.x = self.x
        model.z = self.z
        model.u = self.u
        model.xdot = self.x_dot

        # CasADi expression for implicit dynamics
        model.f_impl_expr = cs.vertcat(self.x_dot - self.rhs,
                                       self.z - self.rhs_impl)
        # CasADi expression for explicit dynamics
        model.f_expl_expr = self.rhs
        
        # model.p = w  # Not used right now
        model.name = "flexible_arm_nq" + str(self.nq)

        return model

    def get_acados_model_safety(self):
        model = AcadosModel()
        model.x = self.x
        model.z = self.z
        model.u = self.u
        model.xdot = self.x_dot
        # model.p = w  # Not used right now

        model.f_impl_expr = cs.vertcat(self.x_dot - self.rhs,
                                       self.z - self.rhs_impl)
        model.f_expl_expr = self.rhs
        
        model.name = "flexible_arm_nq" + str(self.nq)
        constraint_expr = self.rhs_impl  # endeffector position

        return model, constraint_expr

    def output(self, x):
        """ Return the output of the system for a given state.
        Actually it wraps output map. It is needed for simulation API
        """
        return np.array(self.h(x))

    def gravity_torque(self, q):
        """ Returns the gravity torque of the active (controlled) joints
        """
        t_ = np.zeros_like(q)
        return np.array(self.rnea(q, t_, t_))[self.qa_idx,:]

    def __str__(self) -> str:
        return f"3dof symbolic flexible arm model with {self.n_seg} segments"


def get_rest_configuration(qa: np.ndarray, n_seg: int) -> np.ndarray:
    """ Computes the rest configuration of the robot based on 
    rest configuration of the active joints. (It assumes that actuators
    provide enough torque to keep the active joints at a desired position; 
    the goal is to compute rest positions of the passive joints)

    :parameter qa: configration of the active joints
    :parameter n_seg: number of segments
    """
    try:
        qa = qa.reshape((3,))
    except:
        RuntimeError

    if n_seg == 0:
        return qa.reshape((3,1))

    # Path to a folder with model description
    model_folder = 'models/three_dof/' + \
                    n_seg_int2str[n_seg] + '_segments/'

    # Load RNEA function
    rnea = cs.Function.load(model_folder + 'rnea.casadi')

    # Number of joints
    nq = 3 + 2*n_seg
    idxs = set(np.arange(0, nq))
    idxs_a = {0, 1, 2 + n_seg}
    idxs_p = idxs.difference(idxs_a)

    # Casadi symbolic variables for passive joints
    qp_link1 = cs.SX.sym('qp_l1', n_seg)
    qp_link2 = cs.SX.sym('qp_l2', n_seg)
    qp = cs.vertcat(qp_link1, qp_link2)
    q = cs.vertcat(qa[:2], qp_link1, qa[2], qp_link2)
    dq = cs.SX.zeros(nq)
    ddq = cs.SX.zeros(nq)

    # Get expression and function for gravity vector
    g_expr = rnea(q, dq, ddq)
    g_expr_p = g_expr[list(idxs_p)]

    # Load stiffness parameters
    params_file = 'flexibility_params.yml'
    params_path = os.path.join(model_folder, params_file)

    with open(params_path) as f:
        flexibility_params = yaml.safe_load(f)

    K2 = np.diag(flexibility_params['K2'])
    K3 = np.diag(flexibility_params['K3'])
    K = block_diag(K2, K3)

    # Create a function for computing rest positions
    f = cs.Function('f', [qp], [g_expr_p + K @ qp])

    # Create a root finder
    F = cs.rootfinder('F', 'newton', f)
    qp_num = np.array(F(np.zeros(2*n_seg))).squeeze()

    # Form a vector of joint angles
    q = np.zeros(nq)
    q[list(idxs_a)] = qa
    q[list(idxs_p)] = qp_num 

    return q.reshape(-1,1)


def inverse_kinematics_rb(pee: np.ndarray, q_guess: np.ndarray = None):
    """ Numerically computes the inverse kineamtics for the
    zero segment model i.e. for the rigid body approximation

    :parameter pee: the end-effector position
    """
    # Load the forward kinematics function
    model_folder = 'models/three_dof/zero_segments/'
    fkp = cs.Function.load(model_folder + 'fkp.casadi')

    # Create symbolic variables for joint angles and
    # create a function for solving ik
    q = cs.SX.sym('q', 3)
    f = cs.Function ('f', [q], [fkp(q) - pee])

    # Create a root finder
    F = cs.rootfinder('F', 'newton', f)

    # Solve IK problem
    if q_guess is None:
        q_guess = np.array([0, 0.1, -0.1])
        
    q_num = F(q_guess)
    return (np.array(q_num) + np.pi) % (2 * np.pi) - np.pi


if __name__ == "__main__":
    n_seg = 3
    # arm = FlexibleArm(n_seg)
    # q = np.zeros((arm.nq, 1))
    # R, p = arm.fk_ee(q)
    # print(R, p)

    # # arm.visualize(q)

    sarm = SymbolicFlexibleArm3DOF(n_seg)
    # sarm = SymbolicFlexibleArm3DOF(n_seg=3, integrator='collocation')
    # print(sarm)

    n_seg = 0
    model_folder = 'models/three_dof/' + \
                       n_seg_int2str[n_seg] + '_segments/'
    urdf_path = os.path.join(model_folder, 
                'flexible_arm_3dof_' + str(n_seg) + 's.urdf')

    # qa = np.random.randn(3)
    # qa = np.array([np.pi/2, np.pi/10, -np.pi/8])
    qa = np.array([0., 2*np.pi/5, -np.pi/3])
    # qa = np.zeros(3)
    q = get_rest_configuration(qa, n_seg)

    fkp = cs.Function.load(model_folder + 'fkp.casadi')
    pee = np.array(fkp(q))
    q_ik = inverse_kinematics_rb(pee)

    q = np.repeat(q.reshape(1,-1), 50, axis=0)
    

    animator = Panda3dAnimator(urdf_path, 0.01, q).play(3)

