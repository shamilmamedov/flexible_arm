import numpy as np
from estimator import ExtendedKalmanFilter
from flexible_arm_3dof import get_rest_configuration, FlexibleArm3DOF, SymbolicFlexibleArm3DOF
from gym_env import FlexibleArmEnv
from imitator import ImitatorOptions, Imitator, plot_training
from mpc_3dof import Mpc3dofOptions, Mpc3Dof
from imitation_builder import ImitationBuilder_Stabilization, ImitationBuilder_Wall
from evaluator import Evaluator

if __name__ == "__main__":
    RUN_EXPERIMENT = False
    PLOT_TRAINING = False

    experiment_dir = "data/imitation/wall/version1"

    # plot training
    if PLOT_TRAINING:
        plot_training(log_dir=experiment_dir)

    imitation_builder_wall = ImitationBuilder_Wall()
    evaluator = Evaluator(builder=imitation_builder_wall, n_episodes=2, policy_dir=experiment_dir)
    evaluator.evaluate_nn(policy_dir=experiment_dir)
    evaluator.evaluate_expert()
    evaluator.print_result()

    # run simulation
    if RUN_EXPERIMENT:
        imitator, env, controller = ImitationBuilder_Wall().build()
        imitator.render_expert(n_episodes=3, n_replay=1, show_plot=False, seed=1)
        imitator.render_student(n_episodes=10, n_replay=1, show_plot=False, seed=1, policy_dir=experiment_dir)
