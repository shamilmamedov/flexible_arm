from dataclasses import dataclass
import numpy as np
from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
import tempfile
from typing import TYPE_CHECKING, Tuple
from stable_baselines3.common.env_checker import check_env
from gym_env import FlexibleArmEnvOptions, FlexibleArmEnv
from gym_utils import CallableExpert
from mpc_3dof import Mpc3Dof
from flexible_arm_3dof import get_rest_configuration

# Avoid circular imports with type checking
if TYPE_CHECKING:
    from flexible_arm_3dof import FlexibleArm3DOF, SymbolicFlexibleArm3DOF


@dataclass
class ImitatorOptions:
    """
    Options for imitation learning
    """

    def __init__(self, dt: float = 0.01):
        self.environment_options: FlexibleArmEnvOptions = FlexibleArmEnvOptions(dt=dt)
        self.filename: str = "trained_policy"
        self.save_file: bool = True
        self.n_episodes: int = 50000


class Imitator:
    """
    Imitation class, used for learning an MPC policy
    """

    def __init__(self, options: ImitatorOptions, expert_controller: Mpc3Dof, estimator=None):
        self.options = options
        self.expert_controller = expert_controller

        # create and sanity check training environment
        self.env = FlexibleArmEnv(options=options.environment_options, estimator=estimator)
        check_env(self.env)

        # set mpc reference and make baselines expert
        u_ref = np.zeros((3,))
        q_mpc = get_rest_configuration(self.env.xee_final[:,0], expert_controller.options.n_seg)
        dq_mpc = np.zeros_like(q_mpc)
        x_mpc = np.vstack((q_mpc, dq_mpc))
        self.expert_controller.set_reference_point(p_ee_ref=self.env.xee_final, x_ref=x_mpc, u_ref=u_ref)
        self.callable_expert = CallableExpert(expert_controller, observation_space=self.env.observation_space,
                                              action_space=self.env.action_space)

        self.venv = DummyVecEnv([lambda: FlexibleArmEnv(options=options.environment_options, estimator=estimator)])
        self.bc_trainer = bc.BC(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space
        )

    def train(self):
        with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
            print(tmpdir)
            dagger_trainer = SimpleDAggerTrainer(
                venv=self.venv,
                scratch_dir=tmpdir,
                expert_policy=self.callable_expert,
                bc_trainer=self.bc_trainer,
            )
            dagger_trainer.train(self.options.n_episodes, rollout_round_min_episodes=20)

        # Evaluate and save trained policy
        reward, _ = evaluate_policy(dagger_trainer.policy, self.env, 10)
        print("Final reward: {}".format(reward))

        # Save policy
        if self.options.save_file:
            dagger_trainer.policy.save(self.options.filename)
