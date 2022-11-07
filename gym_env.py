"""
Implements a gym environment for flexible link robot arm. The environment
is used for behavioral cloning
"""
from copy import copy
from dataclasses import dataclass

import gym
import numpy as np
from typing import Tuple
from gym import spaces

from flexible_arm_3dof import SymbolicFlexibleArm3DOF, FlexibleArm3DOF, get_rest_configuration
from simulation import Simulator, SimulatorOptions


@dataclass
class FlexibleArmEnvOptions:
    """
    Options for imitation learning
    """
    def __init__(self, dt: float = 0.01):
        self.qa_start: np.ndarray = np.array([1.5, 0.0, 1.5])
        self.qa_end: np.ndarray = np.array([1.5, 0.0, 1.5])
        self.qa_range_start: np.ndarray = np.array([np.pi, np.pi, np.pi])
        self.qa_range_end: np.ndarray = np.array([.0, .0, .0])
        self.n_seg: int = 3
        self.dt: float = dt
        self.render_mode = None
        self.maximum_torques: np.ndarray = np.array([20, 10, 10])
        self.goal_dist_euclid: float = 0.01
        self.sim_time = 3
        self.goal_min_time: float = 1


class FlexibleArmEnv(gym.Env):
    """ Creates gym environment for flexible robot arm

    """

    def __init__(self, options: FlexibleArmEnvOptions, estimator=None) -> None:
        """
        :parameter n_seg: number of segments per link
        :parameter dt: stepsize of the integrator
        :parameter q0: initial rest configuration of the robot
        """
        self.options = options
        self.model = FlexibleArm3DOF(options.n_seg)
        self.model_sym = SymbolicFlexibleArm3DOF(options.n_seg)
        self.dt = options.dt

        # initial position
        self._state, _ = self.sample_rand_config(qa_mean=options.qa_start, qa_range=options.qa_range_start)

        # end position
        self.x_final, self.xee_final = self.sample_rand_config(qa_mean=options.qa_end, qa_range=options.qa_range_end)

        self.no_intg_steps = 0
        self.max_intg_steps = int(options.sim_time / options.dt)

        # Simulator of the model
        if estimator is not None:
            sim_opts = SimulatorOptions(contr_input_states='estimated')
        else:
            sim_opts = SimulatorOptions()
        sim_opts.dt = self.options.dt
        self.simulator = Simulator(self.model_sym, controller=None, integrator='cvodes',
                                   estimator=estimator, opts=sim_opts)

        # Define observation space
        if estimator is None:
            self.observation_space = spaces.Box(np.array([-np.pi * 200] * self.model.nx),
                                                np.array([np.pi * 200] * self.model.nx), dtype=np.float64)
        else:
            self.observation_space = spaces.Box(np.array([-np.pi * 200] * estimator.model.nx),
                                                np.array([np.pi * 200] * estimator.model.nx), dtype=np.float64)

        # Define action space
        self.action_space = spaces.Box(-options.maximum_torques, options.maximum_torques, dtype=np.float64)

        self.render_mode = options.render_mode
        self.goal_dist_counter = 0

        self._state = None

    def sample_rand_config(self, qa_mean: np.ndarray, qa_range: np.ndarray):
        qa = np.random.uniform(-qa_range / 2, qa_range / 2) + qa_mean
        q = get_rest_configuration(qa, self.options.n_seg)
        dq = np.zeros_like(q)
        x = np.vstack((q, dq))
        _, xee = self.model.fk_ee(q)
        return x[:, 0], xee

    def reset(self):
        """
        """
        # Reset state of the robot
        # initial position
        self._state, _ = self.sample_rand_config(qa_mean=self.options.qa_start,
                                                 qa_range=self.options.qa_range_start)
        # end position
        self.x_final, self.xee_final = self.sample_rand_config(qa_mean=self.options.qa_end,
                                                               qa_range=self.options.qa_range_end)

        self.simulator.reset(x0=self._state)

        # Reset integrations step counter
        self.no_intg_steps = 0
        self.goal_dist_counter = 0

        # Get observations and info
        observation = self._state

        return observation

    def step(self, a) -> Tuple[np.ndarray, float, bool, dict]:
        """
        TODO: Make sure actions are in the action space;
              Clip actions if necessary 
        """
        # Take action
        self._state = self.simulator.step(a)
        self.no_intg_steps += 1

        # define reward as Euclidian distance to goal
        _, x_ee = self.model.fk_ee(self._state[:int(self.model.nq)])
        dist = np.linalg.norm(x_ee - self.xee_final, 2)
        reward = - dist * self.options.dt

        # Check if the state is terminal
        done = bool(self._terminal(dist))

        # Other outputs
        info = {}

        return (self._state[:,0], reward, done, info)

    def _terminal(self, dist: float):
        if dist < self.options.goal_dist_euclid:
            self.goal_dist_counter += 1
        else:
            self.goal_dist_counter = 0

        done = False
        if self.goal_dist_counter >= self.options.goal_min_time / self.options.dt:
            done = True
        if self.no_intg_steps > self.max_intg_steps:
            done = True
        return bool(done)
