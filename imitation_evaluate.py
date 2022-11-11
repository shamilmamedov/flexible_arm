from imitator import plot_training
from imitation_builder import ImitationBuilder_Wall, ImitationBuilder_Wall2
from evaluator import Evaluator

if __name__ == "__main__":
    RUN_EXPERIMENT = False
    PLOT_TRAINING = False
    RUN_EVALUATOR = True
    RENDER = False
    SHOW_PLOTS = True

    experiment_dir = "data/imitation/wall2/version2"

    # plot training
    if PLOT_TRAINING:
        plot_training(log_dir=experiment_dir)

    if RUN_EVALUATOR:
        imitation_builder_wall = ImitationBuilder_Wall2()
        evaluator = Evaluator(builder=imitation_builder_wall, n_episodes=1, policy_dir=experiment_dir,
                              render=RENDER, show_plots=SHOW_PLOTS)
        evaluator.evaluate_nn_safe(policy_dir=experiment_dir)
        evaluator.evaluate_expert()
        evaluator.evaluate_nn(policy_dir=experiment_dir)

        evaluator.print_result()
