import numpy as np
import torch
from stable_baselines3.common import policies


from envs.flexible_arm_3dof import SymbolicFlexibleArm3DOF
from envs.flexible_arm_env import (
    FlexibleArmEnv,
    FlexibleArmEnvOptions,
    WallObstacle
)
from mpc_3dof import Mpc3dofOptions, Mpc3Dof
from utils.utils import StateType


class CallableMPCExpert(policies.BasePolicy):
    """a callable expert is needed for sb3 which involves the mpc controler"""

    def __init__(
        self,
        controller,
        observation_space,
        action_space,
        observation_includes_goal: bool,
        observation_includes_obstacle: bool = False,
    ):
        super().__init__(observation_space, action_space)
        self.observation_includes_goal = observation_includes_goal
        self.observation_includes_obstacle = observation_includes_obstacle
        self.controller = controller
        self.observation_space = observation_space
        self.action_space = action_space
  
    def _parse_observation(self, observation: np.ndarray):
        """
        separates the observation into the state, and goal coordinates
        observation: ndarray of shape (batch_size, observation_space.shape[0])
        returns: state, goal_coords, wallObstacle
        """
        if self.observation_includes_obstacle:
            state = observation[:,:-9]
            goal_coords = observation[:,-9:-6]
            w = observation[:,-6:-3].ravel()
            b = observation[:,-3:].ravel()
            return state, goal_coords, WallObstacle(w, b)
        else:
            state = observation[:,:-3]
            goal_coords = observation[:,-3:]
            return state, goal_coords, None

    def _predict(self, observation, deterministic: bool = False):
        """
        predict method used internally in sb3 for learning. needs torch datatyps
        @param observation: observation as the states of the system of SHAPE (batch_size, observation_space.shape[0])
        @param deterministic: Needed by superclass, dummy
        @return: torques computed by expert
        """
        if isinstance(observation, torch.Tensor):
            observation = observation.numpy()
        torques = self.predict_np(observation)
        torques = torch.tensor(torques)
        return torques

    def predict_np(self, observation, deterministic: bool = False):
        """
        Same as other predict method but with numpy datadypes
        @param observation: observation as the states of the system of SHAPE (batch_size, observation_space.shape[0])
        @param deterministic: Needed by superclass, dummy
        @return: torques computed by expert
        """
        B, D = observation.shape
        if self.observation_includes_goal:
            observation, goal_coords, obstacle = self._parse_observation(observation)

            self.controller.set_reference_point(p_ee_ref=goal_coords.reshape(-1, B))
        observation = observation.reshape(-1, B)
        n_q = observation.shape[0] // 2
        torques = self.controller.compute_torques(
            q=observation[0:n_q, :], dq=observation[n_q:, :]
        )
        return torques

    def __call__(self, observation):
        return self._predict(observation)


def create_unified_flexiblearmenv_and_controller(create_controller=False, add_wall_obstacle=False):
    """
    This is to make sure that all algorithms are trained and evaluated on the same environment settings.
    """
    # --- Create FlexibleArm environment ---
    n_seg = 5
    n_seg_mpc = 3

    # Environment options
    R_Q = [3e-6] * 3
    R_DQ = [2e-3] * 3
    R_PEE = [1e-4] * 3
    env_options = FlexibleArmEnvOptions(
        n_seg=n_seg,
        n_seg_estimator=n_seg_mpc,
        sim_time=1.5,
        dt=0.004,
        qa_range_start=np.array([-np.pi/2, 0., -np.pi+0.05]),
        qa_range_end=np.array([3*np.pi/2, np.pi, np.pi-0.05]),
        contr_input_states=StateType.ESTIMATED,  # "real" if the n_seg is the same for the data and control env
        sim_noise_R=np.diag([*R_Q, *R_DQ, *R_PEE]),
        render_mode="human",
    )
    # Wall obstacle
    if add_wall_obstacle:
        # Wall obstacle
        w = np.array([0., 1., 0.])
        b = np.array([0., -0.15, 0.5])
        wall = WallObstacle(w, b)
        observation_includes_obstacle = True
    else:
        wall = None
        observation_includes_obstacle = False

    # Create environment
    env = FlexibleArmEnv(env_options, obstacle=wall)
    # -------------------------------------
    if create_controller:
        # --- Create MPC controller ---
        fa_sym_mpc = SymbolicFlexibleArm3DOF(n_seg_mpc)
        mpc_options = Mpc3dofOptions(n_seg=n_seg_mpc, tf=0.5, n=125)
        controller = Mpc3Dof(model=fa_sym_mpc, x0=None, pee_0=None, options=mpc_options)

        # create MPC expert
        expert = CallableMPCExpert(
            controller,
            observation_space=env.observation_space,
            action_space=env.action_space,
            observation_includes_goal=True,
            observation_includes_obstacle=observation_includes_obstacle
        )
    else:
        expert = None
    return env, expert
