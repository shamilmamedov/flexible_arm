import numpy as np

from flexible_arm_3dof import get_rest_configuration, FlexibleArm3DOF, SymbolicFlexibleArm3DOF
from gym_env import FlexibleArmEnv
from imitator import ImitatorOptions, Imitator
from mpc_3dof import Mpc3dofOptions, Mpc3Dof

if __name__ == "__main__":
    imitator_options = ImitatorOptions(dt=0.01)
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
    estimator = None

    # get reward of expert policy
    env = FlexibleArmEnv(options=imitator_options.environment_options, estimator=estimator)

    # create imitator and train
    imitator = Imitator(options=imitator_options, expert_controller=controller, estimator=estimator)

    # evaluate expert to get an idea of the reward achievable
    imitator.evaluate_expert(n_eps=1)

    # train network to imitate expert
    imitator.train()

    # evaluate
    imitator.evaluate_student()
