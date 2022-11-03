import numpy as np
from estimator import ExtendedKalmanFilter
from flexible_arm_3dof import get_rest_configuration, FlexibleArm3DOF, SymbolicFlexibleArm3DOF
from gym_env import FlexibleArmEnv
from imitator import ImitatorOptions, Imitator
from mpc_3dof import Mpc3dofOptions, Mpc3Dof
from imitation_builder import ImitationBuilder_Stabilization

if __name__ == "__main__":
    EVALUATE_EXPERT = False
    TRAIN_POLICY = True
    imitator, env, controller = ImitationBuilder_Stabilization().build()

    # evaluate expert to get an idea of the reward achievable
    if EVALUATE_EXPERT:
        imitator.evaluate_expert(n_eps=1)

    # train network to imitate expert
    if TRAIN_POLICY:
        imitator.train()
