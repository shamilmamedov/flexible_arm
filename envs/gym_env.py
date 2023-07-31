"""
Implements a gym environment for flexible link robot arm. The environment
is used for behavioral cloning
"""
from typing import Tuple, Optional
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from estimator import ExtendedKalmanFilter
from envs.flexible_arm_3dof import SymbolicFlexibleArm3DOF, get_rest_configuration
from simulation import Simulator, SimulatorOptions
from animation import Panda3dRenderer


@dataclass
class FlexibleArmEnvOptions:
    """
    Options for imitation learning
    """

    dt: float = 0.01
    n_seg: int = 3
    sim_time: float = 3
    qa_start: np.ndarray = np.array([1.5, 0.0, 1.5])
    qa_end: np.ndarray = np.array([1.5, 0.0, 1.5])
    qa_range_start: np.ndarray = np.array([np.pi, np.pi, np.pi])
    qa_range_end: np.ndarray = np.array([0.0, 0.0, 0.0])
    render_mode = None
    maximum_torques: np.ndarray = np.array([20, 10, 10])
    goal_dist_euclid: float = 0.01
    goal_min_time: float = 1
    sim_noise_R: np.ndarray = None
    contr_input_states: str = "estimated"
    render_mode: Optional[str] = None


class FlexibleArmEnv(gym.Env):
    """Creates gym environment for flexible robot arm"""

    def __init__(
        self,
        options: FlexibleArmEnvOptions,
        estimator: ExtendedKalmanFilter = None,
    ) -> None:
        """
        :parameter n_seg: number of segments per link
        :parameter dt: stepsize of the integrator
        :parameter q0: initial rest configuration of the robot
        """
        self.render_mode = options.render_mode
        # if estimator is not None:
        #    assert estimator.x_hat is not None, "Estimator needs to be initialized"

        self.options = options
        self.model_sym = SymbolicFlexibleArm3DOF(options.n_seg)
        self.dt = options.dt

        # initial position
        self._state, _ = self.sample_rand_config(
            qa_mean=options.qa_start, qa_range=options.qa_range_start
        )

        # end position
        self.x_final, self.xee_final = self.sample_rand_config(
            qa_mean=options.qa_end, qa_range=options.qa_range_end
        )

        # counter for integfration steps and max integration steps
        self.no_intg_steps = 0
        self.max_intg_steps = int(options.sim_time / options.dt)

        # Simulator of the model
        sim_opts = SimulatorOptions(
            dt=self.options.dt,
            n_iter=self.max_intg_steps,
            R=self.options.sim_noise_R,
            contr_input_states=self.options.contr_input_states,
        )
        self.simulator = Simulator(
            self.model_sym,
            controller=None,
            integrator="cvodes",
            estimator=estimator,
            opts=sim_opts,
        )
        # Define observation space
        if estimator is None:
            nx_ = self.model_sym.nx
            self.observation_space = spaces.Box(
                np.array([-np.pi * 200] * nx_),
                np.array([np.pi * 200] * nx_),
                dtype=np.float64,
            )
        else:
            self.observation_space = spaces.Box(
                np.array([-np.pi * 200] * estimator.model.nx),
                np.array([np.pi * 200] * estimator.model.nx),
                dtype=np.float64,
            )

        # Define action space
        self.action_space = spaces.Box(
            -options.maximum_torques, options.maximum_torques, dtype=np.float64
        )

        self.goal_dist_counter = 0
        self.stop_if_goal_condition = True

        self._state = None

    def sample_rand_config(self, qa_mean: np.ndarray, qa_range: np.ndarray):
        """Samples a random joint configuration from a given range using
        uniform distrubution
        """
        qa = np.random.uniform(-qa_range / 2, qa_range / 2) + qa_mean
        q = get_rest_configuration(qa, self.options.n_seg)
        dq = np.zeros_like(q)
        x = np.vstack((q, dq))
        xee = np.array(self.model_sym.p_ee(q))
        return x[:, 0], xee

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        # Reset state of the robot
        # initial position
        self._state, _ = self.sample_rand_config(
            qa_mean=self.options.qa_start, qa_range=self.options.qa_range_start
        )
        # end position
        self.x_final, self.xee_final = self.sample_rand_config(
            qa_mean=self.options.qa_end, qa_range=self.options.qa_range_end
        )

        self.simulator.reset(x0=self._state)

        # Reset integrations step counter
        self.no_intg_steps = 0
        self.goal_dist_counter = 0

        # Get observations and info
        if self.options.contr_input_states == "real":
            observation = self._state
        else:
            observation = self.simulator.estimator.x_hat[:, 0]
        return observation, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        TODO: Make sure actions are in the action space;
              Clip actions if necessary
        """
        # TODO ask Rudi why? They should return the same things
        if self.simulator.estimator is None:
            self._state = self.simulator.step(action)
        else:
            self.simulator.step(action)
            self._state = self.simulator.x[self.simulator.k, :]
        self.no_intg_steps += 1

        # define reward as Euclidian distance to goal
        x_ee = np.array(self.model_sym.p_ee(self._state[: int(self.model_sym.nq)]))
        dist = np.linalg.norm(x_ee - self.xee_final, 2)
        reward = -dist * self.options.dt

        # Check if the state is terminal
        terminated = self._terminal(dist)
        truncated = self._truncated()

        # Other outputs
        info = {}

        # Get observations and info
        if self.simulator.estimator is None:
            observation = self._state[:, 0]
        else:
            observation = self.simulator.x_hat[self.no_intg_steps, :]

        return observation, reward, terminated, truncated, info

    def _terminal(self, dist: float):
        if dist < self.options.goal_dist_euclid:
            self.goal_dist_counter += 1
        else:
            self.goal_dist_counter = 0

        done = False
        if (
            self.goal_dist_counter >= self.options.goal_min_time / self.options.dt
        ) and self.stop_if_goal_condition:
            done = True
        return bool(done)

    def _truncated(self):
        return bool(self.no_intg_steps >= self.max_intg_steps)

    def render(self):
        if not self.render_mode in ["human", "rgb_array"]:
            raise ValueError(
                f"Invalid render_mode: {self.render_mode}. Must be one of ['human', 'rgb_array']"
            )

        if not hasattr(self, "renderer"):
            self.renderer = Panda3dRenderer(self.model_sym.urdf_path)
        frame = self.renderer.render(q=self._state[: int(self.model_sym.nq)])
        return frame
