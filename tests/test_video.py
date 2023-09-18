"""
Recording videos for the final report
"""
import os
import sys
import logging

import cv2
import torch
import numpy as np
from hydra import compose, initialize

from stable_baselines3 import SAC
from stable_baselines3 import PPO


from utils.utils import seed_everything
from utils.gym_utils import (
    create_unified_flexiblearmenv_and_controller_and_safety_filter,
    SafetyWrapper,
)


# Get hydra config
initialize(version_base=None, config_path="../conf", job_name="FlexibleArm")
cfg = compose(config_name="config", overrides=sys.argv[1:])
logging.basicConfig(level=logging.INFO)

SEED = cfg.training.seed
DEVICE = cfg.training.device
VIDEO_DIR = "videos"
NUM_EPISODES = 10
seed_everything(SEED)
os.makedirs(VIDEO_DIR, exist_ok=True)


def frames_to_video(frames: np.array, video_name: str = "video.mp4"):
    """
    :param frames: np.array of shape (n_frames, height, width, 3)
    :return:
    """
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_name, fourcc, 30, (width, height))
    for frame in frames:
        video.write(frame)
    video.release()


def run_rollouts_frames(model, reset_states) -> list[np.array]:
    episode_frames = []
    for i, rs in enumerate(reset_states):
        frames = []
        logging.info(f"Episode {i}")
        state, _ = env.reset(seed=SEED, options=rs)
        frame = env.render()
        frames.append(frame)
        done = False
        while not done:
            action, _ = model.predict(state, deterministic=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            frame = env.render()
            frames.append(frame)
            done = terminated or truncated
            state = next_state
        frames = np.array(frames)[..., 0:3]
        episode_frames.append(frames)
    return episode_frames


logging.info("Creating the environment")
(
    env,
    expert,
    safety_filter,
) = create_unified_flexiblearmenv_and_controller_and_safety_filter(
    create_controller=True, add_wall_obstacle=True, create_safety_filter=True
)

# --- collecting reset states ---
logging.info("Collecting reset states")
reset_states = []
for i in range(NUM_EPISODES):
    _, state_dict = env.reset(seed=SEED)
    reset_states.append(state_dict)


# --- record NMPC ---
logging.info("Recording NMPC")
frames = run_rollouts_frames(expert, reset_states)
for i, ep_frames in enumerate(frames):
    frames_to_video(ep_frames, video_name=f"videos/NMPC_{i+1}.mp4")

# --- record DAGGER + SF ---
logging.info("Recording DAGGER+SF")
dagger_model = torch.load(cfg.kpi.dagger_model_path)
dagger_sf_model = SafetyWrapper(policy=dagger_model, safety_filter=safety_filter)
frames = run_rollouts_frames(dagger_sf_model, reset_states)
for i, ep_frames in enumerate(frames):
    frames_to_video(ep_frames, video_name=f"videos/DAGGER_SF_{i+1}.mp4")


# --- record SAC + SF ---
logging.info("Recording SAC+SF")
sac_model = SAC.load(cfg.kpi.sac_model_path)
sac_sf_model = SafetyWrapper(policy=sac_model.policy, safety_filter=safety_filter)
frames = run_rollouts_frames(sac_sf_model, reset_states)
for i, ep_frames in enumerate(frames):
    frames_to_video(ep_frames, video_name=f"videos/SAC_SF_{i+1}.mp4")


# --- record DAGGER ---
logging.info("Recording DAGGER")
frames = run_rollouts_frames(dagger_model, reset_states)
for i, ep_frames in enumerate(frames):
    frames_to_video(ep_frames, video_name=f"videos/DAGGER_{i+1}.mp4")

# --- record SAC ---
logging.info("Recording SAC")
frames = run_rollouts_frames(sac_model, reset_states)
for i, ep_frames in enumerate(frames):
    frames_to_video(ep_frames, video_name=f"videos/SAC_{i+1}.mp4")

# --- record PPO ---
logging.info("Recording PPO")
ppo_model = PPO.load(cfg.kpi.ppo_model_path)
frames = run_rollouts_frames(ppo_model, reset_states)
for i, ep_frames in enumerate(frames):
    frames_to_video(ep_frames, video_name=f"videos/PPO_{i+1}.mp4")

# --- record BC ---
logging.info("Recording BC")
bc_model = torch.load(cfg.kpi.bc_model_path)
frames = run_rollouts_frames(bc_model, reset_states)
for i, ep_frames in enumerate(frames):
    frames_to_video(ep_frames, video_name=f"videos/BC_{i+1}.mp4")

# --- record GAIL ---
logging.info("Recording GAIL")
gail_model = SAC.load(cfg.kpi.gail_model_path)
frames = run_rollouts_frames(gail_model, reset_states)
for i, ep_frames in enumerate(frames):
    frames_to_video(ep_frames, video_name=f"videos/GAIL_{i+1}.mp4")

# --- record AIRL ---
logging.info("Recording AIRL")
airl_model = SAC.load(cfg.kpi.airl_model_path)
frames = run_rollouts_frames(airl_model, reset_states)
for i, ep_frames in enumerate(frames):
    frames_to_video(ep_frames, video_name=f"videos/AIRL_{i+1}.mp4")

# --- record DENSITY ---
logging.info("Recording DENSITY")
density_model = SAC.load(cfg.kpi.density_model_path)
frames = run_rollouts_frames(density_model, reset_states)
for i, ep_frames in enumerate(frames):
    frames_to_video(ep_frames, video_name=f"videos/DENSITY_{i+1}.mp4")
