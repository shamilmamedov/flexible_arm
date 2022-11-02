import numpy as np

from estimator import ExtendedKalmanFilter
from flexible_arm_3dof import get_rest_configuration, FlexibleArm3DOF, SymbolicFlexibleArm3DOF
from gym_env import FlexibleArmEnv
from imitator import ImitatorOptions, Imitator
from mpc_3dof import Mpc3dofOptions, Mpc3Dof

if __name__ == "__main__":
    USE_ESTIMATOR = True
    TRAIN_POLICY = False

    dt = 0.01
    imitator_options = ImitatorOptions(dt=dt)
    imitator_options.environment_options.sim_time = 4
    imitator_options.environment_options.n_seg = 3

    # Create FlexibleArm instances
    n_seg_mpc = 3
    fa_sym_ld = SymbolicFlexibleArm3DOF(n_seg_mpc)

    # Create mpc options and controller
    mpc_options = Mpc3dofOptions(n_seg=n_seg_mpc, tf=0.3)
    mpc_options.n = 30
    controller = Mpc3Dof(model=fa_sym_ld, options=mpc_options)

    # Create estimator
    est_model = SymbolicFlexibleArm3DOF(n_seg=3, dt=dt)
    P0 = 0.01 * np.ones((est_model.nx, est_model.nx))
    q_q, q_dq = [1e-2] * est_model.nq, [1e-1] * est_model.nq
    Q = np.diag([*q_q, *q_dq])
    r_q, r_dq, r_pee = [3e-4] * 3, [6e-2] * 3, [1e-2] * 3
    R = np.diag([*r_q, *r_dq, *r_pee])
    if USE_ESTIMATOR:
        estimator = ExtendedKalmanFilter(est_model, None, P0, Q, R)
    else:
        estimator = None

    # get reward of expert policy
    env = FlexibleArmEnv(options=imitator_options.environment_options, estimator=estimator)

    # create imitator and train
    imitator = Imitator(options=imitator_options, expert_controller=controller, estimator=estimator)

    # evaluate expert to get an idea of the reward achievable
    imitator.evaluate_expert(n_eps=1)

    # train network to imitate expert
    if TRAIN_POLICY:
        imitator.train()

    # evaluate
    imitator.evaluate_student(n_episodes=10, n_replay=2, show_plot=False)
