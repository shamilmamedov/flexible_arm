"""
Implements a gym environment for flexible link robot arm. The environment
is used for behavioral cloning
"""

import gym
import numpy as np
from typing import Tuple
from gym import spaces

from flexible_arm_3dof import SymbolicFlexibleArm3DOF, FlexibleArm3DOF


class FlexibleArmEnv(gym.Env):
    """ Creates gym environment for flexible robot arm
    """

    def __init__(self, n_seg, ts, render_mode=None) -> None:
        """
        :parameter n_seg: number of segments per link
        :parameter ts: stepsize of the integrator
        """
        self.ts = ts
        self.n_intg_steps = 0
        self.max_intg_steps = 500

        # Numerical arm model (an alternative is symbolic)
        self.model = FlexibleArm3DOF(n_seg)

        # Define observation space
        x_low = np.array([-np.finfo(np.float32).max]*self.model.nx)
        x_high = np.array([np.finfo(np.float32).max]*self.model.nx)
        self.observation_space = spaces.Box(x_low, x_high, dtype=np.float32)

        # Define action space
        u_low = np.array([-100]*3)
        u_high = np.array([100]*3)
        self.action_space = spaces.Box(u_low, u_high, dtype = np.float32)



if __name__ == "__main__":
    print('Trying to create gym environment')
