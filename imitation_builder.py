from abc import ABC, abstractmethod
import numpy as np
from estimator import ExtendedKalmanFilter
from flexible_arm_3dof import get_rest_configuration, FlexibleArm3DOF, SymbolicFlexibleArm3DOF
from gym_env import FlexibleArmEnv
from imitator import ImitatorOptions, Imitator
from mpc_3dof import Mpc3dofOptions, Mpc3Dof
from safty_filter_3dof import SafetyFilter3dofOptions, SafetyFilter3Dof


class ImitationBuilder(ABC):
    def __init__(self, n_seg: int, n_seg_mpc: int, n_seg_safety: int, dt: float, tf: float, dir_rel: str):
        self.imitator_options = ImitatorOptions(dt=dt)
        self.imitator_options.environment_options.n_seg = n_seg
        self.imitator_options.environment_options.dt = dt
        self.imitator_options.logdir_rel = "/data/imitation" + dir_rel
        self.dt = dt
        # Create FlexibleArm instances
        self.fa_sym_ld = SymbolicFlexibleArm3DOF(n_seg_mpc)

        # Create mpc options and controller
        self.mpc_options = Mpc3dofOptions(n_seg=n_seg_mpc, tf=tf)

        # create safety filter
        self.safety_options = SafetyFilter3dofOptions(n_seg=n_seg_safety)
        self.estimator = None

    @abstractmethod
    def pre_build(self):
        pass

    def build(self):
        self.pre_build()

        # get reward of expert policy
        env = FlexibleArmEnv(options=self.imitator_options.environment_options, estimator=self.estimator)

        # create safety filter
        fa_sym_ld = SymbolicFlexibleArm3DOF(self.safety_options.n_seg)
        fa_sym = SymbolicFlexibleArm3DOF(self.safety_options.n_seg)
        safety_filter = SafetyFilter3Dof(model=fa_sym_ld, model_nonsymbolic=fa_sym,
                                         options=self.safety_options)

        # construct controller:
        controller = Mpc3Dof(model=self.fa_sym_ld, options=self.mpc_options)

        # create imitator and train
        imitator = Imitator(options=self.imitator_options, expert_controller=controller, estimator=self.estimator)

        return imitator, env, controller, safety_filter


class ImitationBuilder_Stabilization(ImitationBuilder):
    def __init__(self):
        n_seg = 3
        n_seg_mpc = 3
        n_seg_safety = 1
        dt = 0.01
        tf = 0.3
        dir_rel = "/stabilization"
        super().__init__(n_seg=n_seg, n_seg_mpc=n_seg_mpc, n_seg_safety=n_seg_safety, dt=dt, tf=tf, dir_rel=dir_rel)

    def pre_build(self):
        self.imitator_options.environment_options.qa_start = np.array([1.5, 0.0, 1.5])
        self.imitator_options.environment_options.qa_end = np.array([1.5, 0.0, 1.5])
        self.imitator_options.environment_options.qa_range_start = np.array([np.pi, np.pi, np.pi])
        self.imitator_options.environment_options.qa_range_end = np.array([.0, .0, .0])
        self.imitator_options.environment_options.n_seg = 3
        self.imitator_options.environment_options.render_mode = None
        self.imitator_options.environment_options.maximum_torques = np.array([20, 10, 10])
        self.imitator_options.environment_options.goal_dist_euclid = 0.01
        self.imitator_options.environment_options.sim_time = 3
        self.imitator_options.environment_options.goal_min_time = 1

        self.imitator_options.n_episodes = 60 * 1000  # number of training episodes (1000 ~ 1 minute on laptop)
        self.imitator_options.rollout_round_min_episodes = 5  # option for dagger algorithm.
        self.imitator_options.rollout_round_min_timesteps = 2000  # option for dagger algorithm.

        self.mpc_options.n = 30  # number of discretization points
        self.mpc_options.nlp_iter = 50  # number of iterations of the nonlinear solver
        self.mpc_options.condensing_relative = 1  # relative factor of condensing [0-1]
        self.mpc_options.wall_constraint_on = False  # choose whether we activate the wall constraint
        self.mpc_options.wall_axis = 2  # Wall axis: 0,1,2 -> x,y,z
        self.mpc_options.wall_value = 0.5  # wall height value on axis
        self.mpc_options.wall_pos_side = True  # defines the allowed side of the wall

        self.safety_options.n = 10  # number of discretization points
        self.safety_options.tf = 0.1  # time horizon
        self.safety_options.nlp_iter = 100  # number of iterations of the nonlinear solver
        self.safety_options.z_diag = np.array([0] * 3) * 1e1
        self.safety_options.z_e_diag = np.array([0] * 3) * 1e3
        self.safety_options.r_diag = np.array([1., 1., 1.]) * 1e1
        self.safety_options.w2_slack_speed = 1e3
        self.safety_options.w2_slack_wall = 1e5
        self.safety_options.wall_constraint_on = self.mpc_options.wall_constraint_on  # choose whether we activate the wall constraint
        self.safety_options.wall_axis = self.mpc_options.wall_axis  # Wall axis: 0,1,2 -> x,y,z
        self.safety_options.wall_value = self.mpc_options.wall_value  # wall height value on axis
        self.safety_options.wall_pos_side = self.mpc_options.wall_pos_side  # defines the allowed side of the wall

        est_model = SymbolicFlexibleArm3DOF(n_seg=self.mpc_options.n_seg, dt=self.dt, integrator='collocation')
        p0_q = [0.05] * est_model.nq  # 0.05, 0.1
        p0_dq = [1e-3] * est_model.nq
        P0 = np.diag([*p0_q, *p0_dq])
        # process noise covariance
        q_q = [1e-4, *[1e-3] * (est_model.nq - 1)]
        q_dq = [1e-1, *[5e-1] * (est_model.nq - 1)]  # 5e-1, 10e-1
        Q = np.diag([*q_q, *q_dq])
        # measurement noise covaiance
        r_q, r_dq, r_pee = [3e-5] * 3, [5e-2] * 3, [1e-3] * 3
        R = 10 * np.diag([*r_q, *r_dq, *r_pee])
        self.imitator_options.environment_options.sim_noise_R = R
        self.estimator = ExtendedKalmanFilter(est_model, None, P0, Q, R)


class ImitationBuilder_Wall(ImitationBuilder):
    def __init__(self):
        n_seg = 3
        n_seg_mpc = 3
        n_seg_safety = 1
        dt = 0.01
        tf = 0.3
        dir_rel = "/wall"
        super().__init__(n_seg=n_seg, n_seg_mpc=n_seg_mpc, n_seg_safety=n_seg_safety, dt=dt, tf=tf, dir_rel=dir_rel)

    def pre_build(self):
        delta_wall_angle = np.pi / 40
        self.imitator_options.environment_options.qa_start = \
            np.array([-np.pi / 2 - delta_wall_angle, np.pi / 7, -np.pi / 5])
        self.imitator_options.environment_options.qa_end = \
            np.array([-delta_wall_angle, np.pi / 7, -np.pi / 5])
        self.imitator_options.environment_options.qa_range_start = np.array([np.pi, np.pi / 2, np.pi / 2])
        self.imitator_options.environment_options.qa_range_end = np.array([.0, .0, .0])
        self.imitator_options.environment_options.render_mode = None
        self.imitator_options.environment_options.maximum_torques = np.array([20, 10, 10])
        self.imitator_options.environment_options.goal_dist_euclid = 0.01
        self.imitator_options.environment_options.sim_time = 3
        self.imitator_options.environment_options.goal_min_time = 1

        self.imitator_options.n_episodes = 100 * 1000  # number of training episodes (1000 ~ 1 minute on laptop)
        self.imitator_options.rollout_round_min_episodes = 10  # option for dagger algorithm.
        self.imitator_options.rollout_round_min_timesteps = 5000  # option for dagger algorithm.

        self.mpc_options.n = 30  # number of discretization points
        self.mpc_options.nlp_iter = 50  # number of iterations of the nonlinear solver
        self.mpc_options.condensing_relative = 1  # relative factor of condensing [0-1]
        self.mpc_options.wall_constraint_on = True  # choose whether we activate the wall constraint
        self.mpc_options.wall_axis = 1  # Wall axis: 0,1,2 -> x,y,z
        self.mpc_options.wall_value = 0.0  # wall height value on axis
        self.mpc_options.wall_pos_side = False  # defines the allowed side of the wall

        self.safety_options.n = 20  # number of discretization points
        self.safety_options.tf = 0.20  # time horizon
        self.safety_options.nlp_iter = 100  # number of iterations of the nonlinear solver
        self.safety_options.z_diag = np.array([0] * 3)
        self.safety_options.z_e_diag = np.array([0] * 3)
        self.safety_options.r_diag = np.array([1., 1., 1.]) * 1e2
        self.safety_options.w_reg_dq = 0.01
        self.safety_options.w_reg_dq_terminal = 1e-2
        self.safety_options.w2_slack_speed = 1e3
        self.safety_options.w2_slack_wall = 3e5
        self.safety_options.wall_constraint_on = self.mpc_options.wall_constraint_on  # choose whether we activate the wall constraint
        self.safety_options.wall_axis = self.mpc_options.wall_axis  # Wall axis: 0,1,2 -> x,y,z
        self.safety_options.wall_value = self.mpc_options.wall_value  # wall height value on axis
        self.safety_options.wall_pos_side = self.mpc_options.wall_pos_side  # defines the allowed side of the wall

        est_model = SymbolicFlexibleArm3DOF(n_seg=self.mpc_options.n_seg, dt=self.dt, integrator='collocation')
        p0_q = [0.05] * est_model.nq  # 0.05, 0.1
        p0_dq = [1e-3] * est_model.nq
        P0 = np.diag([*p0_q, *p0_dq])
        # process noise covariance
        q_q = [1e-4, *[1e-3] * (est_model.nq - 1)]
        q_dq = [1e-1, *[5e-1] * (est_model.nq - 1)]  # 5e-1, 10e-1
        Q = np.diag([*q_q, *q_dq])
        # measurement noise covaiance
        r_q, r_dq, r_pee = [3e-5] * 3, [5e-2] * 3, [1e-3] * 3
        R = 10 * np.diag([*r_q, *r_dq, *r_pee])

        self.imitator_options.environment_options.sim_noise_R = R/5
        self.estimator = ExtendedKalmanFilter(est_model, None, P0, Q, R)


class ImitationBuilder_Wall2(ImitationBuilder):
    def __init__(self):
        n_seg = 10
        n_seg_mpc = 2
        n_seg_safety = 1
        dt = 0.01
        tf = 0.3
        dir_rel = "/wall2"
        super().__init__(n_seg=n_seg, n_seg_mpc=n_seg_mpc, n_seg_safety=n_seg_safety, dt=dt, tf=tf, dir_rel=dir_rel)

    def pre_build(self):
        delta_wall_angle = np.pi / 40
        self.imitator_options.environment_options.qa_start = \
            np.array([-np.pi / 4 - delta_wall_angle, np.pi / 7, -np.pi / 5])
        self.imitator_options.environment_options.qa_end = \
            np.array([-delta_wall_angle, np.pi / 7, -np.pi / 5])
        self.imitator_options.environment_options.qa_range_start = np.array([np.pi/2, np.pi / 4, np.pi / 4])
        self.imitator_options.environment_options.qa_range_end = np.array([.0, .0, .0])
        self.imitator_options.environment_options.render_mode = None
        self.imitator_options.environment_options.maximum_torques = np.array([20, 10, 10])
        self.imitator_options.environment_options.goal_dist_euclid = 0.01
        self.imitator_options.environment_options.sim_time = 4
        self.imitator_options.environment_options.goal_min_time = 1.5

        self.imitator_options.n_episodes = 100 * 1000  # number of training episodes (1000 ~ 1 minute on laptop)
        self.imitator_options.rollout_round_min_episodes = 10  # option for dagger algorithm.
        self.imitator_options.rollout_round_min_timesteps = 5000  # option for dagger algorithm.

        self.mpc_options.n = 30  # number of discretization points
        self.mpc_options.nlp_iter = 50  # number of iterations of the nonlinear solver
        self.mpc_options.condensing_relative = 1  # relative factor of condensing [0-1]
        self.mpc_options.w2_slack_speed = 1e6
        self.mpc_options.wall_constraint_on = True  # choose whether we activate the wall constraint
        self.mpc_options.wall_axis = 1  # Wall axis: 0,1,2 -> x,y,z
        self.mpc_options.wall_value = 0.0  # wall height value on axis
        self.mpc_options.wall_pos_side = False  # defines the allowed side of the wall

        self.safety_options.n = 20  # number of discretization points
        self.safety_options.tf = 0.20  # time horizon
        self.safety_options.nlp_iter = 100  # number of iterations of the nonlinear solver
        self.safety_options.z_diag = np.array([0] * 3)
        self.safety_options.z_e_diag = np.array([0] * 3)
        self.safety_options.r_diag = np.array([1., 1., 1.])*1
        self.safety_options.r_diag_rollout = np.array([1., 1., 1.]) * 1e-5
        self.safety_options.w_reg_dq = 0.01
        self.safety_options.w_reg_dq_terminal = 1e-2
        self.safety_options.w2_slack_speed = 1e6
        self.safety_options.w2_slack_wall = 3e5
        self.safety_options.w1_slack_wall = 1e6
        self.safety_options.wall_constraint_on = self.mpc_options.wall_constraint_on  # choose whether we activate the wall constraint
        self.safety_options.wall_axis = self.mpc_options.wall_axis  # Wall axis: 0,1,2 -> x,y,z
        self.safety_options.wall_value = self.mpc_options.wall_value  # wall height value on axis
        self.safety_options.wall_pos_side = self.mpc_options.wall_pos_side  # defines the allowed side of the wall

        est_model = SymbolicFlexibleArm3DOF(n_seg=self.mpc_options.n_seg, dt=self.dt, integrator='collocation')
        p0_q = [0.05] * est_model.nq  # 0.05, 0.1
        p0_dq = [1e-3] * est_model.nq
        P0 = np.diag([*p0_q, *p0_dq])
        # process noise covariance
        q_q = [1e-4, *[1e-3] * (est_model.nq - 1)]
        q_dq = [1e-1, *[5e-1] * (est_model.nq - 1)]  # 5e-1, 10e-1
        Q = np.diag([*q_q, *q_dq])
        # measurement noise covaiance
        r_q, r_dq, r_pee = [3e-5] * 3, [5e-2] * 3, [1e-3] * 3
        R = 10 * np.diag([*r_q, *r_dq, *r_pee])

        self.imitator_options.environment_options.sim_noise_R = R/20
        self.estimator = ExtendedKalmanFilter(est_model, None, P0, Q, R)


class ImitationBuilder_Wall3(ImitationBuilder):
    def __init__(self):
        n_seg = 10
        n_seg_mpc = 2
        n_seg_safety = 1
        dt = 0.01
        tf = 0.5
        dir_rel = "/wall3"
        super().__init__(n_seg=n_seg, n_seg_mpc=n_seg_mpc, n_seg_safety=n_seg_safety, dt=dt, tf=tf, dir_rel=dir_rel)

    def pre_build(self):
        delta_wall_angle = np.pi / 40
        self.imitator_options.environment_options.qa_start = \
            np.array([-np.pi / 4 - delta_wall_angle, np.pi / 7, -np.pi / 5])
        self.imitator_options.environment_options.qa_end = \
            np.array([-delta_wall_angle, np.pi / 7, -np.pi / 5])
        self.imitator_options.environment_options.qa_range_start = np.array([np.pi/2, np.pi / 4, np.pi / 4])
        self.imitator_options.environment_options.qa_range_end = np.array([.0, .0, .0])
        self.imitator_options.environment_options.render_mode = None
        self.imitator_options.environment_options.maximum_torques = np.array([20, 10, 10])
        self.imitator_options.environment_options.goal_dist_euclid = 0.01
        self.imitator_options.environment_options.sim_time = 3
        self.imitator_options.environment_options.goal_min_time = 1.5

        self.imitator_options.n_episodes = 40 * 1000  # number of training episodes (1000 ~ 1 minute on laptop)
        self.imitator_options.rollout_round_min_episodes = 10  # option for dagger algorithm.
        self.imitator_options.rollout_round_min_timesteps = 5000  # option for dagger algorithm.

        self.mpc_options.n = 50  # number of discretization points
        self.mpc_options.nlp_iter = 5  # number of iterations of the nonlinear solver
        self.mpc_options.condensing_relative = 1  # relative factor of condensing [0-1]
        self.mpc_options.w2_slack_speed = 1e6
        self.mpc_options.wall_constraint_on = True  # choose whether we activate the wall constraint
        self.mpc_options.wall_axis = 1  # Wall axis: 0,1,2 -> x,y,z
        self.mpc_options.wall_value = 0.0  # wall height value on axis
        self.mpc_options.wall_pos_side = False  # defines the allowed side of the wall

        self.safety_options.n = 20  # number of discretization points
        self.safety_options.tf = 0.20  # time horizon
        self.safety_options.nlp_iter = 100  # number of iterations of the nonlinear solver
        self.safety_options.z_diag = np.array([0] * 3)
        self.safety_options.z_e_diag = np.array([0] * 3)
        self.safety_options.r_diag = np.array([1., 1., 1.])*1
        self.safety_options.r_diag_rollout = np.array([1., 1., 1.]) * 1e-5
        self.safety_options.w_reg_dq = 0.01
        self.safety_options.w_reg_dq_terminal = 1e-2
        self.safety_options.w2_slack_speed = 1e3
        self.safety_options.w2_slack_wall = 3e5
        self.safety_options.w1_slack_wall = 1e6
        self.safety_options.wall_constraint_on = self.mpc_options.wall_constraint_on  # choose whether we activate the wall constraint
        self.safety_options.wall_axis = self.mpc_options.wall_axis  # Wall axis: 0,1,2 -> x,y,z
        self.safety_options.wall_value = self.mpc_options.wall_value  # wall height value on axis
        self.safety_options.wall_pos_side = self.mpc_options.wall_pos_side  # defines the allowed side of the wall

        est_model = SymbolicFlexibleArm3DOF(n_seg=self.mpc_options.n_seg, dt=self.dt, integrator='collocation')
        p0_q = [0.05] * est_model.nq  # 0.05, 0.1
        p0_dq = [1e-3] * est_model.nq
        P0 = np.diag([*p0_q, *p0_dq])
        # process noise covariance
        q_q = [1e-4, *[1e-3] * (est_model.nq - 1)]
        q_dq = [1e-1, *[5e-1] * (est_model.nq - 1)]  # 5e-1, 10e-1
        Q = np.diag([*q_q, *q_dq])
        # measurement noise covaiance
        r_q, r_dq, r_pee = [3e-5] * 3, [5e-2] * 3, [1e-3] * 3
        R = 100 * np.diag([*r_q, *r_dq, *r_pee])

        self.imitator_options.environment_options.sim_noise_R = R/600
        self.estimator = ExtendedKalmanFilter(est_model, None, P0, Q, R)