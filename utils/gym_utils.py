from typing import Dict
import logging

import numpy as np
import torch
from stable_baselines3.common import policies
import torch

from envs.flexible_arm_3dof import SymbolicFlexibleArm3DOF
from envs.flexible_arm_env import FlexibleArmEnv, FlexibleArmEnvOptions, WallObstacle
from mpc_3dof import Mpc3dofOptions, Mpc3Dof
from safty_filter_3dof import SafetyFilter3dofOptions, SafetyFilter3Dof
from utils.utils import StateType

logging.basicConfig(level=logging.INFO)


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
            state = observation[:, :-9]  # state includes [q, dq, p_ee]
            goal_coords = observation[:, -9:-6]
            w = observation[:, -6:-3].ravel()
            b = observation[:, -3:].ravel()
            return state, goal_coords, WallObstacle(w, b)
        else:
            state = observation[:, :-3]
            goal_coords = observation[:, -3:]
            return state, goal_coords, None

    def _predict(self, observation, deterministic: bool = False):
        """
        predict method used internally in sb3 for learning. needs torch datatyps
        @param observation: observation as the states of the system of SHAPE (batch_size, observation_space.shape[0])
        @param deterministic: Needed by superclass, dummy
        @return: torques computed by expert
        """
        if isinstance(observation, torch.Tensor):
            observation = observation.cpu().numpy()
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
            if obstacle is not None:
                self.controller.set_wall_parameters(w=obstacle.w, b=obstacle.b)
        observation = observation.reshape(-1, B)

        # observation entries: [states_q, states_dq, p_ee]
        n_q = (observation.shape[0] - 3) // 2
        try:
            torques = self.controller.compute_torques(
                q=observation[0:n_q, :], dq=observation[n_q : 2 * n_q, :]
            )
        except Exception as e:
            logging.warning(f"Exception in MPC controller: {e}")
            logging.warning("Setting torques to zero")
            torques = np.zeros(self.action_space.shape)
        return torques

    def __call__(self, observation):
        return self._predict(observation)


class SafetyWrapper(policies.BasePolicy):
    """
    This is a wrapper class that takes in a policy and adds a safety filter and modifies the
    unsafe actions of the policy if need be.
    """

    def __init__(self, policy: policies.BasePolicy, safety_filter: SafetyFilter3Dof):
        super().__init__(policy.observation_space, policy.action_space)
        self.policy = policy
        self.safety_filter = safety_filter
        self.observation_includes_obstacle = safety_filter.options.wall_constraint_on
        self.observation_includes_goal = True

    def _parse_observation(self, observation: np.ndarray):
        """
        separates the observation into the state, and goal coordinates
        observation: ndarray of shape (batch_size, observation_space.shape[0])
        returns: state, goal_coords, wallObstacle
        """
        if self.observation_includes_obstacle:
            state = observation[:, :-9]
            goal_coords = observation[:, -9:-6]
            w = observation[:, -6:-3].ravel()
            b = observation[:, -3:].ravel()
            return state, goal_coords, WallObstacle(w, b)
        else:
            state = observation[:, :-3]
            goal_coords = observation[:, -3:]
            return state, goal_coords, None

    def forward(self, observation: torch.Tensor, deterministic: bool = False):
        return self._predict(observation, deterministic)

    def _predict(
        self, observation: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        with torch.no_grad():
            proposed_action = self.policy._predict(observation, deterministic)

        safe_action = self._apply_safety(proposed_action, observation)
        return safe_action

    def _apply_safety(self, proposed_action: torch.Tensor, observation: torch.Tensor):
        """
        Same as _predict but with the proposed action as input
        """
        if isinstance(observation, torch.Tensor):
            observation = observation.cpu().numpy()

        B, D = observation.shape
        if self.observation_includes_goal:
            observation, goal_coords, obstacle = self._parse_observation(observation)

            # self.safety_filter.set_reference_point(p_ee_ref=goal_coords.reshape(-1, B))
            if obstacle is not None:
                self.safety_filter.set_wall_parameters(w=obstacle.w, b=obstacle.b)

        if isinstance(proposed_action, torch.Tensor):
            proposed_action = proposed_action.cpu().numpy()

        # get pee coordinate
        # todo verify this is true
        # in order to extract [qa, dqa and pee of the observation, we need to know the number of segments
        n_seg = ((observation.shape[1] - 3) - 6) // 4
        qa_idx = [0, 1, 2 + n_seg]
        d_offset = 1 + 2 * (n_seg + 1)
        dqa_idx = [d_offset, d_offset + 1, d_offset + 2 + n_seg]
        qa = observation[0, qa_idx]
        dqa = observation[0, dqa_idx]
        p_ee = observation[0, -3:]
        y = np.hstack((qa, dqa, p_ee))
        safe_action = self.safety_filter.filter(u0=proposed_action, y=y)
        # print(np.max((safe_action-proposed_action)))
        safe_action = torch.from_numpy(safe_action)
        return safe_action


def create_unified_flexiblearmenv_and_controller_and_safety_filter(
    create_controller=False,
    create_safety_filter=False,
    add_wall_obstacle=False,
    env_opts: Dict = None,
    cntrl_opts: Dict = None,
    safety_fltr_opts: Dict = None,
):
    """
    This is to make sure that all algorithms are trained and evaluated on the same environment settings.
    @env_opts: dict valued environment options are overwritten
    @cntrl_opts: dict valued mpc options are overwritten
    @safety_fltr_opts: dict valued safety filter options are overwritten
    """
    # --- Create FlexibleArm environment ---
    n_seg = 10
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
        goal_min_time=0.01,
        qa_range_start=np.array([-np.pi / 2, 0.0, -np.pi + 0.075]),
        qa_range_end=np.array([3 * np.pi / 2, np.pi, np.pi - 0.075]),
        contr_input_states=StateType.ESTIMATED,  # "real" if the n_seg is the same for the data and control env
        sim_noise_R=np.diag([*R_Q, *R_DQ, *R_PEE]),
        render_mode="human",
    )

    # set other options of environment, wich are passed as dictionary
    if env_opts:
        env_options.update(env_opts)

    # Wall obstacle
    if add_wall_obstacle:
        # Wall obstacle
        w = np.array([0.0, 1.0, 0.0])
        b = np.array([0.0, -0.15, 0.5])
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

        # set other options of the controller, wich are passed as dictionary
        if cntrl_opts:
            mpc_options.update(cntrl_opts)

        controller = Mpc3Dof(model=fa_sym_mpc, x0=None, pee_0=None, options=mpc_options)

        # create MPC expert
        expert = CallableMPCExpert(
            controller,
            observation_space=env.observation_space,
            action_space=env.action_space,
            observation_includes_goal=True,
            observation_includes_obstacle=observation_includes_obstacle,
        )
    else:
        expert = None

    if create_safety_filter:
        safety_filter_options = SafetyFilter3dofOptions()
        # set other options of safety filter, wich are passed as dictionary
        if safety_fltr_opts:
            safety_filter_options.update(safety_fltr_opts)

        safety_filter = SafetyFilter3Dof(options=safety_filter_options)
    else:
        safety_filter = None
    return env, expert, safety_filter
