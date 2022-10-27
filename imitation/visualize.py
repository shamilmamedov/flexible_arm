import argparse
from pathlib import Path
from logging import getLogger

import gym
import numpy as np
from imitation.imitation_agent import ImitationAgent
import racecars.env  # noqa: F401
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


logger = getLogger(__name__)


def load_net_and_env(root: Path):
    cfg = OmegaConf.load(root / ".hydra/config.yaml")
    cfg.device = "cpu"

    # Create the env.
    env_name = f"rc-{cfg.data.env}-v0"
    print(f"The name of the env: {env_name}")
    env = gym.make(env_name)

    # Recreate the net.
    ckpt_dict = torch.load(root / "imitation_model.ckpt")
    net_state_dict = ckpt_dict["net"]

    net = instantiate(cfg.net, **ckpt_dict["kw"])

    net.load_state_dict(net_state_dict)

    return net, env


def visualize(args):
    root = Path(args.path)
    save_video_path = (
        Path(args.save_video_path) if args.save_video_path is not None else None
    )

    cfg = OmegaConf.load(root / ".hydra/config.yaml")
    cfg.device = "cpu"
    cfg.exp_dir = args.path

    net, env = load_net_and_env(root)

    default_mpc_param = env.ego_agent.default_action
    # default_mpc_param = env.ego_agent.default_action
    env.unwrapped.ego_agent = ImitationAgent(env.ego_agent, net, use_mpc_freq=None)

    print(env.observation_space)
    # Enjoy trained agent

    for episode_idx in range(args.num_episodes):
        obs = env.reset()
        done = False

        while not done:
            print(f"Real obs: {np.array(obs)}")
            obs, reward, done, _ = env.step(default_mpc_param)
            print("Reward: {}".format(reward))

        ani = env.render_new()

        if save_video_path is not None:
            # Save episodes as gifs.
            Path(save_video_path).mkdir(parents=True, exist_ok=True)
            ani.save(save_video_path / f"recorded_episode_{episode_idx}.gif")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visualizes the results.")
    parser.add_argument("path", type=str)
    parser.add_argument("--save-video-path", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=10 * 15 * 10)
    args = parser.parse_args()

    visualize(args)
