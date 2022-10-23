import numpy as np
import torch
from stable_baselines3.common import policies


class CallableExpert(policies.BasePolicy):
    """ a callable expert is needed for sb3 which involves the mpc controler"""

    def __init__(self, controller, observation_space, action_space):
        super().__init__(observation_space, action_space)
        self.controller = controller
        self.observation_space = observation_space
        self.action_space = action_space

    def _predict(self, observation, deterministic: bool = False):
        """
        predict method used internally in sb3 for learning. needs torch datatyps
        @param observation: observation as the states of the system
        @param deterministic: Needed by superclass, dummy
        @return: torques computed by expert
        """
        n_q = int(observation.tolist()[0].__len__() / 2)
        torques = self.controller.compute_torques(np.array([observation.tolist()[0][:n_q]]).transpose(),
                                                  np.array([observation.tolist()[0][n_q:]]).transpose())
        torques = torch.tensor(torques)
        return torques

    def predict_np(self, observation, deterministic: bool = False):
        """
        Same as other predict method but with numpy datadypes
        @param observation: observation as the states of the system
        @param deterministic: Needed by superclass, dummy
        @return: torques computed by expert
        """
        nq = int(observation.__len__() / 2)
        torques = self.controller.compute_torques(np.expand_dims(observation[:nq], axis=1),
                                                  np.expand_dims(observation[nq:], axis=1))
        return torques

    def __call__(self, observation):
        return self._predict(observation)
