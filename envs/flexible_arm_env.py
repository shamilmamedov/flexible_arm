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
from utils.utils import StateType


# dataclass that stores the wall obstacle parameters defined
# using hyperplane equation w.T @ x >= b
@dataclass
class WallObstacle:
    w: np.ndarray  # 3x1 vector
    b: np.ndarray  # 3x1 vector


@dataclass
class FlexibleArmEnvOptions:
    """
    Options for imitation learning

    dt: stepsize of the integrator
    n_seg: number of segments per link
    n_seg_estimator: number of segments for the estimator model (this should match the MPC model's n_seg)

    NOTE: MultiTask Learning (changing the goal) is possible by setting
    qa_range_end to values other than 0.
    """

    dt: float = 0.01
    n_seg: int = 3
    n_seg_estimator: int = 3
    sim_time: float = 3
    qa_range_start: np.ndarray = np.array([-np.pi / 2, 0.0, -np.pi + 0.05])
    qa_range_end: np.ndarray = np.array([3 * np.pi / 2, np.pi, np.pi - 0.05])
    render_mode = None
    maximum_torques: np.ndarray = np.array([20, 10, 10])
    goal_dist_euclid: float = 0.01
    goal_min_time: float = 0.03
    sim_noise_R: np.ndarray = None
    contr_input_states: StateType = StateType.REAL
    render_mode: Optional[str] = None


class FlexibleArmEnv(gym.Env):
    """
    Creates gym environment for flexible robot arm.
    NOTE:Observation space currently consists of: [current_state, goal_state, goal_position].
    Including the goal in the observation space, allows for multi-task learning.
    If multi-task learning is not desired, set qa_range_end to all zeros and only
    use the (L-3)/2 elements of the observation (L: length of the observation vector)
    """

    def __init__(
        self, options: FlexibleArmEnvOptions, obstacle: Optional[WallObstacle] = None
    ) -> None:
        self.options = options
        self.model_sym = SymbolicFlexibleArm3DOF(options.n_seg)
        self.dt = options.dt
        self.obstacle = obstacle

        # counter for integfration steps and max integration steps
        self.no_intg_steps = 0
        self.max_intg_steps = int(options.sim_time / options.dt)

        # Create an estimator if needed
        if options.contr_input_states is StateType.ESTIMATED:
            estimator = self._create_estimator(
                n_seg_estimator=options.n_seg_estimator, dt=options.dt
            )
        else:
            estimator = None

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
        nx_ = (
            estimator.model.nx
            if options.contr_input_states is StateType.ESTIMATED
            else self.model_sym.nx
        )
        nx_ += 3  # add goal position dimensions
        if self.obstacle is not None:
            nx_ += 6  # add obstacle dimensions

        self.observation_space = spaces.Box(
            np.array([-np.pi * 200] * nx_),
            np.array([np.pi * 200] * nx_),
            dtype=np.float64,
        )
        # Define action space
        self.action_space = spaces.Box(
            -options.maximum_torques, options.maximum_torques, dtype=np.float64
        )

        self.goal_dist_counter = 0
        self.stop_if_goal_condition = True
        self._state = None

        # self.render_mode needs to exist because of gymnasium
        self.render_mode = options.render_mode
        # if self.render_mode in ["human", "rgb_array"]:
        #     self.renderer = Panda3dRenderer(self.model_sym.urdf_path)
        # else:
        #     self.renderer = None

    def _create_estimator(
        self, n_seg_estimator: int, dt: float
    ) -> ExtendedKalmanFilter:
        estimator_model = SymbolicFlexibleArm3DOF(
            n_seg_estimator, dt=dt, integrator="cvodes"
        )
        p0_q, p0_dq = [0.05] * estimator_model.nq, [1e-3] * estimator_model.nq
        P0 = np.diag([*p0_q, *p0_dq])
        q_q = [1e-4, *[1e-3] * (estimator_model.nq - 1)]
        q_dq = [1e-1, *[5e-1] * (estimator_model.nq - 1)]
        Q = np.diag([*q_q, *q_dq])
        r_q, r_dq, r_pee = [3e-5] * 3, [5e-2] * 3, [1e-3] * 3
        R = 10 * np.diag([*r_q, *r_dq, *r_pee])
        zero_x0 = np.zeros(((3 + 2 * n_seg_estimator) * 2, 1))
        estimator = ExtendedKalmanFilter(estimator_model, zero_x0, P0, Q, R)
        return estimator

    def sample_rand_config(self, use_estimator: bool = False):
        """
        Samples a random joint configuration from a given range using
        uniform distrubution
        :param use_estimator: if True, the estimator model is used for the number of segments and
                            the forward kinematics (end effector position)
        :return: (sampled active & passive joint configs and their derivatives, end effector position)
        """
        _dinstance_margin = 0.01
        n_seg = self.options.n_seg_estimator if use_estimator else self.options.n_seg
        model = self.simulator.estimator.model if use_estimator else self.model_sym
        while True:
            qa = np.random.uniform(
                self.options.qa_range_start, self.options.qa_range_end
            )
            q = get_rest_configuration(qa, n_seg)
            p_ee = np.array(model.p_ee(q))
            p_elbow = np.array(model.p_elbow(q))

            wall_pos = -0.15 + _dinstance_margin
            if p_ee[2] > 0 and p_ee[1] > wall_pos and p_elbow[1] > wall_pos:
                break

        dq = np.zeros_like(q)
        x = np.vstack((q, dq))
        return x[:, 0], p_ee

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Reset state of the robot
        # initial position
        self._state, _ = self.sample_rand_config(
            use_estimator=False,
        )

        # end position
        # NOTE: If the estimator is availabe we use the dimention of the estimator
        # otherwise we need to use the dimention of the model and then estimate the goal using the estimator.
        # Estimating the goal seems unecessary since we are the ones deciding where to go.
        use_estimator = self.options.contr_input_states is StateType.ESTIMATED
        self.x_final, self.xee_final = self.sample_rand_config(
            use_estimator=use_estimator,
        )

        self.simulator.reset(x0=self._state)  # also estimates the current state

        # Draw the goal position if the gui is available
        if hasattr(self, "renderer"):
            self.renderer.draw_sphere(pos=self.xee_final)

        # Reset integrations step counter
        self.no_intg_steps = 0
        self.goal_dist_counter = 0

        # Get observations and info
        if self.options.contr_input_states is StateType.ESTIMATED:
            observation = self.simulator.estimator.x_hat[:, 0]
        else:
            observation = self._state

        # Add goal position to observation
        observation = np.hstack((observation, self.xee_final.flatten()))
        if self.obstacle is not None:
            observation = np.hstack((observation, self.obstacle.w, self.obstacle.b))
        return observation, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        TODO: Make sure actions are in the action space;
              Clip actions if necessary
        """
        self.simulator.step(action)
        # Real state (not estimated)
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
        if self.options.contr_input_states is StateType.ESTIMATED:
            observation = self.simulator.x_hat[self.no_intg_steps, :]
        else:
            observation = self._state[:, 0]

        # Add goal position to observation
        observation = np.hstack((observation, self.xee_final.flatten()))
        if self.obstacle is not None:
            observation = np.hstack((observation, self.obstacle.w, self.obstacle.b))
        return observation, reward, terminated, truncated, info

    def _terminal(self, dist: float):
        if dist < self.options.goal_dist_euclid:
            self.goal_dist_counter += 1
        else:
            self.goal_dist_counter = 0

        done = False
        if (
            self.goal_dist_counter >= int(self.options.goal_min_time / self.options.dt)
        ) and self.stop_if_goal_condition:
            done = True
        return bool(done)

    def _truncated(self):
        return bool(self.no_intg_steps >= self.max_intg_steps)

    def render(self):
        if self.render_mode in ["human", "rgb_array"]:
            if not hasattr(self, "renderer"):
                self.renderer = Panda3dRenderer(self.model_sym.urdf_path)
                self.renderer.draw_sphere(pos=self.xee_final)
            frame = self.renderer.render(q=self._state[: int(self.model_sym.nq)])
        else:
            raise ValueError(
                "No renderer defined. Render mode can be one of ['human', 'rgb_array']"
            )
        return frame
