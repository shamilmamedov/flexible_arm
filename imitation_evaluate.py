import numpy as np
from estimator import ExtendedKalmanFilter
from flexible_arm_3dof import get_rest_configuration, FlexibleArm3DOF, SymbolicFlexibleArm3DOF
from gym_env import FlexibleArmEnv
from imitator import ImitatorOptions, Imitator, plot_training
from mpc_3dof import Mpc3dofOptions, Mpc3Dof
from imitation_builder import ImitationBuilder_Stabilization, ImitationBuilder_Wall

if __name__ == "__main__":
    RUN_EXPERIMENT = True

    experiment_dir = "data/imitation/wall/version1"

    # plot training
    plot_training(log_dir=experiment_dir)

    # run simulation
    if RUN_EXPERIMENT:
        imitator, env, controller = ImitationBuilder_Wall().build()
        imitator.render_expert(n_episodes=1, n_replay=2, show_plot=False, seed=1)
        imitator.render_student(n_episodes=10, n_replay=2, show_plot=False, seed=1, policy_dir=experiment_dir)






