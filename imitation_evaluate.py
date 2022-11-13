from imitator import plot_training
from imitation_builder import ImitationBuilder_Wall, ImitationBuilder_Wall2, ImitationBuilder_Wall3
from evaluator import Evaluator

if __name__ == "__main__":
    PLOT_TRAINING = False
    RUN_EVALUATOR = True
    PERFORM_RUNS = True
    RENDER = False
    SHOW_PLOTS = False

    experiment_dir = "data/imitation/wall3/version1"

    # plot training
    if PLOT_TRAINING:
        plot_training(log_dir=experiment_dir)

    if RUN_EVALUATOR:
        imitation_builder_wall = ImitationBuilder_Wall3()
        evaluator = Evaluator(builder=imitation_builder_wall, n_episodes=20, policy_dir=experiment_dir,
                              render=RENDER, show_plots=SHOW_PLOTS, n_mpc=[10, 20, 40, 80])
        if PERFORM_RUNS:
            evaluator.evaluate_all(policy_dir=experiment_dir)
        evaluator.print_all()
