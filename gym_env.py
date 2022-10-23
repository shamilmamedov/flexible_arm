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

    def __init__(self, n_seg: int, dt: float, q0, xee_final, render_mode=None) -> None:
        """
        :parameter n_seg: number of segments per link
        :parameter dt: stepsize of the integrator
        :parameter q0: initial rest configuration of the robot
        """
        if len(q0.shape) > 1:
            q0 = q0[:, 0]
        self.q0 = q0
        self.dt = dt
        self.xee_final = xee_final
        self.no_intg_steps = 0
        self.max_intg_steps = 30

        # Numerical arm model (an alternative is symbolic)
        self.model = FlexibleArm3DOF(n_seg)

        # Simulator of the model
        self.simulator = Simulator(self.model, controller=None, integrator='RK45',
                                   estimator=None, )

        # Define observation space
        # todo shamil: this should be our actual maxima of the states (degerees etc)
        self.observation_space = spaces.Box(np.array([-np.pi * 200] * self.model.nx),
                                            np.array([np.pi * 200] * self.model.nx), dtype=np.float64)

        # Define action space
        # todo: think of propper bounds also for mpc (tech specification of robot)
        u_low = np.array([-10] * 3)
        u_high = np.array([10] * 3)
        self.action_space = spaces.Box(u_low, u_high, dtype=np.float64)

        self.render_mode = render_mode

        self._state = None

    def reset(self):
        """
        """
        # Reset state of the robot
        dq = np.zeros(self.model.nq)
        self._state = np.concatenate((self.q0, dq))

        # Reset integrations step counter
        self.no_intg_steps = 0

        # Get observations and info
        observation = self._state

        return observation

    def step(self, a) -> Tuple[np.ndarray, float, bool, dict]:
        """
        TODO: Make sure actions are in the action space;
              Clip actions if necessary 
        """
        s = self._state
        assert s is not None, "Call reset before using object."

        # Take action
        self._state = self.simulator.step(s, a, self.dt)
        self.no_intg_steps += 1

        # define reward as Euclidian distance to goal
        _, x_ee = self.model.fk_ee(self._state[:int(self.model.nq)])
        dist = np.linalg.norm(x_ee - self.xee_final, 2)
        reward = - np.linalg.norm(x_ee - self.xee_final, 2)

        # Check if the state is terminal
        done = self._terminal(dist)

        # Other outputs
        info = {}

        return (self._state, reward, done, info)

    def _terminal(self, dist: float):
        return self.no_intg_steps > self.max_intg_steps or dist < 0.01

