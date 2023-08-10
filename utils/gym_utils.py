import numpy as np
import torch
from stable_baselines3.common import policies


class CallableMPCExpert(policies.BasePolicy):
    """a callable expert is needed for sb3 which involves the mpc controler"""

    def __init__(
        self,
        controller,
        observation_space,
        action_space,
        observation_includes_goal: bool,
    ):
        super().__init__(observation_space, action_space)
        self.observation_includes_goal = observation_includes_goal
        self.controller = controller
        self.observation_space = observation_space
        self.action_space = action_space

    def _separate_observation_and_goal(self, observation: np.ndarray):
        """
        separates the observation into the state, goal state and goal coordinates
        observation: ndarray of shape (batch_size, observation_space.shape[0])
        returns: state, goal_state, goal_coords
        """
        L = self.observation_space.shape[0]
        state = observation[:, 0 : int((L - 3) / 2)]
        goal_state = observation[:, int((L - 3) / 2) : L - 3]
        goal_coords = observation[:, L - 3 :]
        return state, goal_state, goal_coords

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
            observation, goal_state, goal_coords = self._separate_observation_and_goal(
                observation
            )
            u_ref = np.zeros((self.action_space.shape[0], 1))
            self.controller.set_reference_point(
                x_ref=goal_state.reshape(-1, B),
                p_ee_ref=goal_coords.reshape(-1, B),
                u_ref=u_ref,
            )
        observation = observation.reshape(-1, B)
        n_q = observation.shape[0] // 2
        torques = self.controller.compute_torques(
            q=observation[0:n_q, :], dq=observation[n_q:, :]
        )
        return torques

    def __call__(self, observation):
        return self._predict(observation)
