"""
Implements a gym environment for flexible link robot arm. The environment
is used for behavioral cloning
"""

import gym
import numpy as np
from typing import Tuple
from gym import spaces

from flexible_arm_3dof import SymbolicFlexibleArm3DOF, FlexibleArm3DOF
from simulation import Simulator


class FlexibleArmEnv(gym.Env):
    """ Creates gym environment for flexible robot arm

    """

    def __init__(self, n_seg, dt, render_mode=None) -> None:
        """
        :parameter n_seg: number of segments per link
        :parameter dt: stepsize of the integrator
        """
        self.dt = dt
        self.n_intg_steps = 0
        self.max_intg_steps = 500

        # Numerical arm model (an alternative is symbolic)
        self.model = FlexibleArm3DOF(n_seg)

        # Simulator of the model
        self.simulator = Simulator(self.model, controller=None, integrator='RK45',
                                   estimator=None,)

        # Define observation space
        x_low = np.array([-np.finfo(np.float32).max]*self.model.nx)
        x_high = np.array([np.finfo(np.float32).max]*self.model.nx)
        self.observation_space = spaces.Box(x_low, x_high, dtype=np.float32)

        # Define action space
        u_low = np.array([-100]*3)
        u_high = np.array([100]*3)
        self.action_space = spaces.Box(u_low, u_high, dtype=np.float32)

        self.render_mode = render_mode

        self._state = None

    def reset(self):
        """
        TODO: Right now positions of the passive joints are zero. 
              Meaning that the sytstem is not at rest. To solve it
              it is necessary to solve root finding problem where 
              positions of the active joints are fixed and positions
              of the passive joints are found based on stiffness matrix
        """
        # Reset state of the robot
        q = self.model.random_q()
        dq = np.zeros(self.model.nq)
        self._state = np.concatenate((q, dq))

        # Reset integrations step counter
        self.no_intg_steps = 0

        # Get observations and info
        observation = self._state
        info = dict()

        return observation, info

    def step(self, a) -> Tuple(np.ndarray, float, bool, bool, dict):
        """
        TODO: Make sure actions are in the action space;
              Clip actions if necessary 
        """
        s = self._state
        assert s is not None, "Call reset before using AcrobotEnv object."

        # Take action
        self._state = self.simulator.step(s, a, self.dt)

        # Check if the state is terminal
        terminated = self._terminal()

        # Compute reward
        reward = -1.0 if not terminated else 0.0

        truncated = False
        info = {}

        return (self._state, reward, terminated, truncated, info)

    def _terminal(self):
        pass


if __name__ == "__main__":
    env = FlexibleArmEnv(n_seg=1, dt=0.01)
    env.reset()
