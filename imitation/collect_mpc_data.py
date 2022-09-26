from collections import defaultdict, deque
from multiprocessing import Pool
from pathlib import Path
import pickle
from dataclasses import dataclass
import random
from typing import Optional, Sequence, Union
from argparse import ArgumentParser
from itertools import product

import gym
import numpy as np
from racecars.agents.mpcrl import MpcrlAgent
from tqdm import tqdm

from imitation.nets import ParamMPCNet

# from imitation.imitation_agent import ImitationAgent


@dataclass
class RecordedTransitions:
    controls: np.ndarray
    inputs: np.ndarray
    observations: np.ndarray
    mpc_params: np.ndarray

    @classmethod
    def stack(cls, records: Sequence["RecordedTransitions"]):
        return cls(
            **{
                k: np.concatenate([getattr(c, k) for c in records])
                for k in cls.__dataclass_fields__
            }
        )

    def __post_init__(self):
        lengths = [len(getattr(self, f)) for f in self.__dataclass_fields__]

        assert all(l == l_next for l, l_next in zip(lengths, lengths[1:]))

    def save(self, filename: str):
        with open(filename, "wb") as file:  # Overwrites any existing file.
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename: str):
        with open(filename, "rb") as file:
            return pickle.load(file)


def create_default_policy(default_action, action_space: gym.Space):
    def policy_fn(obs):
        return default_action

    return policy_fn


def create_random_policy(default_action, action_space: gym.Space):
    def policy_fn(obs):
        return action_space.sample()

    return policy_fn


def create_mixed_random_policy(default_action, action_space: gym.Space):
    action_queue = deque(maxlen=10)
    default_prob = 0.5
    repeat_action_prob = 0.4

    def policy_fn(obs):
        if not action_queue:
            action = (
                default_action
                if random.random() < default_prob
                else action_space.sample()
            )
            num_repititions = np.random.geometric(repeat_action_prob)
            action_queue.extend([action] * num_repititions)

        return action_queue.pop()

    return policy_fn


def record_mpc_agent_episode(
    seed: int,
    env_name: str,
    mpc_param_policy: Union[str, Path] = "random",
    imitation_net: Optional[ParamMPCNet] = None,
    use_mpc_prob: float = 0.0,
):
    env = gym.make(id=f"rc-{env_name}-v0")

    ego_agent: MpcrlAgent = env.ego_agent
    default_mpc_param = ego_agent.default_action

    if isinstance(mpc_param_policy, Path):
        raise NotImplementedError
    elif mpc_param_policy == "random":
        mpc_param_policy = create_random_policy(default_mpc_param, env.action_space)
    elif mpc_param_policy == "default":
        mpc_param_policy = create_default_policy(default_mpc_param, env.action_space)
    elif mpc_param_policy == "mixed":
        mpc_param_policy = create_mixed_random_policy(
            default_mpc_param, env.action_space
        )
    else:
        raise NotImplementedError

    # imitation_agent = ImitationAgent(
    #     env.agent, imitation_net, use_mpc_prob=use_mpc_prob
    # )

    data = defaultdict(list)

    env.seed(seed)
    obs = env.reset()
    data["observations"].append(obs)

    done = False

    while not done:
        mpc_param = mpc_param_policy(obs)

        obs, _, done, _ = env.step(mpc_param)

        if not done:
            planning_data = ego_agent.planner.get_formatted_solution(t0=0)
            data["controls"].append(planning_data.u)
            data["inputs"].append(planning_data.x)
            data["mpc_params"].append(mpc_param)
            data["observations"].append(obs)

    del data["observations"][-1]

    if len(data["observations"]) == 0:
        return None

    return RecordedTransitions(**{k: np.stack(v) for k, v in data.items()})


def mp_recording(
    record_func, func_args, min_num_samples: int = int(5e4), num_workers: int = 25
):
    chunks = []

    num_samples = 0
    sample_idx = 0

    def args_with_seed():
        start = num_workers * sample_idx
        end = start + num_workers
        return [(seed, *func_args) for seed in range(start, end)]

    with Pool(num_workers) as pool:
        with tqdm(total=min_num_samples) as pbar:

            while num_samples < min_num_samples:
                new_chunks = pool.starmap(record_func, args_with_seed())
                new_chunks = [c for c in new_chunks if c is not None]
                chunks.extend(new_chunks)

                num_samples_new = sum([len(c.controls) for c in new_chunks])
                num_samples += num_samples_new
                pbar.update(num_samples_new)

                sample_idx += 1

            pbar.close()

    return RecordedTransitions.stack(chunks)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--envs", default="mpcrl_lead", type=str)
    parser.add_argument("--num-samples", default=50000, type=int)
    parser.add_argument("--num-workers", default=25, type=int)
    parser.add_argument("--sampling", default="default,mixed", type=str)

    args = parser.parse_args()

    for scenario, mpc_param_policy in product(
        args.envs.split(","), args.sampling.split(",")
    ):
        func_args = [scenario, mpc_param_policy]

        transitions = mp_recording(
            record_mpc_agent_episode, func_args, args.num_samples, args.num_workers
        )

        transitions.save(
            f"scenario_{scenario}_n_{args.num_samples}_s_{mpc_param_policy}.pkl"
        )
