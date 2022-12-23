from imitator import plot_training, plot4paper
from imitation_builder import ImitationBuilder_Wall, ImitationBuilder_Wall2, ImitationBuilder_Wall3
from evaluator import Evaluator

if __name__ == "__main__":
    PLOT_TRAINING = True
    RUN_EVALUATOR = False
    PERFORM_RUNS = False
    PLOT_LATEX = True
    RENDER = True
    SHOW_PLOTS = False

    experiment_dir = "data/imitation/wall3/version1"
    experiment_dir2 = "data/imitation/wall2/version2"

    # plot training
    if PLOT_TRAINING:
        plot4paper(log_dir=experiment_dir2, n_smooth=5, n_max=140)
        plot_training(log_dir=experiment_dir2, n_smooth=5)

    if RUN_EVALUATOR:
        imitation_builder_wall = ImitationBuilder_Wall3()
        evaluator = Evaluator(builder=imitation_builder_wall, n_episodes=1, policy_dir=experiment_dir,
                              render=RENDER, show_plots=SHOW_PLOTS, n_mpc=[10, 20, 40, 80])
        if PLOT_LATEX:
            evaluator.plot_eval_run(policy_dir=experiment_dir, seed=9, load_last_run=True)
        if PERFORM_RUNS:
            evaluator.evaluate_all(policy_dir=experiment_dir, save_file_name='expert_eval_tmp.txt', seed=5)
        evaluator.print_all()
