"""
Implements a gym environment for flexible link robot arm. The environment
is used for behavioral cloning
"""
import logging
from typing import Tuple, Optional
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from estimator import ExtendedKalmanFilter
from envs.flexible_arm_3dof import SymbolicFlexibleArm3DOF, get_rest_configuration
from simulation import Simulator, SimulatorOptions
from animation import Panda3dRenderer
from utils.utils import StateType, Updatable

logging.basicConfig(level=logging.INFO)


# dataclass that stores the wall obstacle parameters defined
# using hyperplane equation w.T @ x >= b
@dataclass
class WallObstacle(Updatable):
    w: np.ndarray  # 3x1 vector
    b: np.ndarray  # 3x1 vector


@dataclass
class FlexibleArmEnvOptions(Updatable):
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
    qa_goal_start: np.ndarray = None
    qa_goal_end: np.ndarray = None
    render_mode = None
    maximum_torques: np.ndarray = np.array([20, 10, 10])
    goal_dist_euclid: float = 0.01
    goal_min_time: float = 0.03
    sim_noise_R: np.ndarray = None
    contr_input_states: StateType = StateType.REAL
    render_mode: Optional[str] = None
    flex_param_file_path: Optional[str] = None


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
        self.model_sym = SymbolicFlexibleArm3DOF(
            options.n_seg, fparams_path=options.flex_param_file_path
        )
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
        nx_ += 6  # add goal and ee position dimensions
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

    def sample_rand_config(
        self,
        use_estimator: bool = False,
        qa_range_start: np.ndarray = None,
        qa_range_end: np.ndarray = None,
        consider_wall: bool = True,
    ):
        """
        Samples a random joint configuration from a given range using
        uniform distrubution
        :param use_estimator: if True, the estimator model is used for the number of segments and
                            the forward kinematics (end effector position)
        :return: (sampled active & passive joint configs and their derivatives, end effector position)
        """
        if qa_range_start is None or qa_range_end is None:
            qa_range_end = self.options.qa_range_end
            qa_range_start = self.options.qa_range_start
        _dinstance_margin = 0.01
        n_seg = self.options.n_seg_estimator if use_estimator else self.options.n_seg
        model = self.simulator.estimator.model if use_estimator else self.model_sym
        while True:
            qa = np.random.uniform(qa_range_start, qa_range_end)
            q = get_rest_configuration(qa, n_seg)
            p_ee = np.array(model.p_ee(q))
            p_elbow = np.array(model.p_elbow(q))

            if not consider_wall:
                break
            wall_pos = -0.15 + _dinstance_margin
            if p_ee[2] > 0 and p_ee[1] > wall_pos and p_elbow[1] > wall_pos:
                break

        dq = np.zeros_like(q)
        x = np.vstack((q, dq))
        return x[:, 0], p_ee

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        # Reset state of the robot
        if options is not None and "state" in options:
            self._state = options["state"]
        else:
            # initial position
            self._state, _ = self.sample_rand_config(
                use_estimator=False, consider_wall=True
            )

        if options is not None and "goal_pos" in options:
            self.xee_final = options["goal_pos"]
        else:
            # end position
            # NOTE: If the estimator is availabe we use the dimention of the estimator
            # otherwise we need to use the dimention of the model and then estimate the goal using the estimator.
            # Estimating the goal seems unecessary since we are the ones deciding where to go.
            # the goal position is either sampled randomly within the range or at a specific point
            use_estimator = self.options.contr_input_states is StateType.ESTIMATED
            _, self.xee_final = self.sample_rand_config(
                use_estimator=use_estimator,
                qa_range_start=self.options.qa_goal_start,
                qa_range_end=self.options.qa_goal_end,
                consider_wall=True,
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

        # compute current end-effector position
        p_ee = np.array(self.model_sym.p_ee(self._state[: int(self.model_sym.nq)]))

        # Add goal position to observation

        if self.obstacle is not None:
            observation = np.hstack(
                (
                    observation,
                    p_ee.flatten(),
                    self.xee_final.flatten(),
                    self.obstacle.w,
                    self.obstacle.b,
                )
            )
        else:
            observation = np.hstack(
                (observation, p_ee.flatten(), self.xee_final.flatten())
            )
        return observation, {"state": self._state, "goal_pos": self.xee_final}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        TODO: Make sure actions are in the action space;
              Clip actions if necessary
        """
        try:
            self.simulator.step(action)
        except RuntimeError as e:
            logging.warning(
                f"Exception in simulator.step: {e}. Simulating with zero action"
            )
            action = np.zeros(self.action_space.shape)
            self.simulator.step(action)

        # Real state (not estimated)
        self._state = self.simulator.x[self.simulator.k, :]
        self.no_intg_steps += 1

        # define reward as Euclidian distance to goal
        p_ee = np.array(self.model_sym.p_ee(self._state[: int(self.model_sym.nq)]))
        p_elb = np.array(self.model_sym.p_elbow(self._state[: int(self.model_sym.nq)]))
        dist = np.linalg.norm(p_ee - self.xee_final, 2)
        reward = -dist * self.options.dt
        
        # Add penalty if the end effector is inside the wall
        if self.obstacle is not None:
            dist_ee2wall = np.dot(self.obstacle.w, p_ee.ravel() - self.obstacle.b)
            dist_elb2wall = np.dot(self.obstacle.w, p_elb.ravel() - self.obstacle.b)
            if dist_ee2wall < 0 or dist_elb2wall < 0:
                reward += -10

        # Add penalty if the end effector is below the ground
        dist_ee2ground = p_ee[2]
        dist_elb2ground = p_elb[2]
        if dist_ee2ground < 0 or dist_elb2ground < 0:
            reward += -10

        # Add penalty if the joint velocities are too high
        qa = self._state[self.model_sym.qa_idx]
        if np.any(np.abs(qa) > self.model_sym.dqa_max):
            reward += -5

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

        if self.obstacle is not None:
            observation = np.hstack(
                (
                    observation,
                    p_ee.flatten(),
                    self.xee_final.flatten(),
                    self.obstacle.w,
                    self.obstacle.b,
                )
            )
        else:
            observation = np.hstack(
                (observation, p_ee.flatten(), self.xee_final.flatten())
            )
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
