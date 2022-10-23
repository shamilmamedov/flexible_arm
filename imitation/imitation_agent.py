"""An MPCRL agent where the MPC part is replaced by an policy learned via imitation learning."""
from collections import deque
import random
from typing import Sequence, Tuple, Optional
from logging import getLogger

import gym
import numpy as np
import torch
from imitation.nets import ParamMPCNet
from racecars.agents.base import AgentStatus
from racecars.agents.mpcrl import MpcrlAgent
from racecars.agents.plain import PlainAgent, PlainAgentParameters
from vehiclegym.automotive_datastructures import FrenetTrajectory
from vehiclegym.road import Road


logger = getLogger(__name__)


class ImitationAgent(PlainAgent):
    def __init__(
        self,
        ego_agent: MpcrlAgent,
        imitation_net: ParamMPCNet,
        use_mpc_freq: Optional[int] = None,
        use_mpc_prob: float = 0.0,
        frame_stacking: int = 5,
    ):
        if isinstance(ego_agent, MpcrlAgent):
            params = PlainAgentParameters(
                ego_agent.params_.ego_model_parameters, check_feasibility=False
            )
            disc_time = ego_agent.planner.time_disc
        else:
            raise NotImplementedError

        observer = ego_agent.observer
        initial_state = FrenetTrajectory(ego_agent.current_state)
        road = ego_agent.road

        self.ego_agent = ego_agent
        self.frame_stacking = frame_stacking
        self.frames = deque(maxlen=frame_stacking)

        super().__init__(observer, initial_state, road, disc_time, params, None)

        self.n_steps = ego_agent.params_.n_simulated_steps
        self.imitation_net = imitation_net
        self.use_mpc_freq = use_mpc_freq
        self.use_mpc_prob = use_mpc_prob
        self.sampled_mpc = False

        self.step_count = 0

        imitation_net.train(False)

    def setup(self, initial_state: FrenetTrajectory, road: Road):
        self.ego_agent.setup(initial_state, road)
        self.step_count = 0
        self.sampled_mpc = False

        self.frames = None

        super().setup(initial_state, road)

    def action_space(self) -> gym.Space:
        return self.ego_agent.action_space

    def default_action(self):
        return self.ego_agent.default_action

    def is_mpc_step(self):
        self.sampled_mpc = True if random.random() < self.use_mpc_prob else False

        if self.use_mpc_freq is None and not self.sampled_mpc:
            return False

        return True if self.step_count % self.use_mpc_freq == 0 else False

    def set_time(self, time: float):
        self.ego_agent.current_time = time
        self.current_time = time
        self.simulator.time_current = time

    def step(
        self,
        action,
        other_cars: Sequence[FrenetTrajectory],
        use_mpc: bool = False,
    ) -> Tuple[FrenetTrajectory, float, AgentStatus]:
        if self.frames is None:
            self.frames = deque(
                [self.observation(other_cars)] * self.frame_stacking,
                maxlen=self.frame_stacking,
            )

        if self.is_mpc_step() or use_mpc:
            self.step_count += 1
            traj, curr_time, state = self.ego_agent.step(action, other_cars)

            self.frames.append(self.observation(other_cars=other_cars))
            # Synchronize time between agents.
            self.set_time(curr_time)
            logger.debug(f"State of MPC step: {state}")

            return traj, curr_time, state

        self.step_count += 1

        obs = torch.tensor(np.stack(self.frames)).unsqueeze(0).to(torch.float32)
        mpc_param = torch.tensor(action).unsqueeze(0).to(torch.float32)

        controls = self.imitation_net(obs, mpc_param).detach().squeeze().numpy().T
        assert len(controls) == self.n_steps

        for control in controls:
            control *= np.array([10000, 1.0])
            traj, curr_time, state = super().step(control, other_cars)

        self.set_time(curr_time)
        self.frames.append(self.observation(other_cars=other_cars))

        logger.debug(f"State of Imitation Net: {state}")

        return traj, curr_time, state
