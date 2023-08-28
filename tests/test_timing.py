"""
This file contains the code for timing the:
    - Decision making (MPC)
    - Inference Time (RL/IRL agent via Neural Network)
"""

import torch
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy
from utils.gym_utils import (
    create_unified_flexiblearmenv_and_controller_and_safety_filter,
)
import matplotlib.pyplot as plt

# --- Neural Network ---
device = torch.device("cuda")
env, _, _ = create_unified_flexiblearmenv_and_controller_and_safety_filter(
    create_controller=False, add_wall_obstacle=True, create_safety_filter=False
)
agent = SAC(policy=SACPolicy, env=env, verbose=0, seed=0)
policy = agent.policy.to(device)
dummy_observations = torch.randn(1, agent.observation_space.shape[0]).to(
    device
)  # single observation

starter = torch.cuda.Event(enable_timing=True)
ender = torch.cuda.Event(enable_timing=True)
reps = 1000
timings = torch.zeros((reps, 1))
# warmup
for _ in range(100):
    _ = policy(dummy_observations)
# timing
with torch.no_grad():
    for i in range(reps):
        starter.record()
        _ = policy(dummy_observations)
        ender.record()
        # synchronize
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[i] = curr_time
nn_mean = np.mean(timings.numpy())
nn_std = np.std(timings.numpy())
print(f"Mean inference time: {nn_mean} ms")
print(f"Std inference time: {nn_std} ms")


# --- MPC ---
# TODO: add MPC timing

# --- Plotting ---
fig, ax = plt.subplots()
ax.bar(
    0,
    nn_mean,
    yerr=nn_std,
    align="center",
    alpha=0.5,
    ecolor="black",
    capsize=10,
    label="NeuralNet",
)
ax.bar(
    1,
    nn_mean,  # TODO: change to MPC mean
    yerr=nn_std,  # TODO: change to MPC std
    align="center",
    alpha=0.5,
    ecolor="black",
    capsize=10,
    label="MPC",
)
ax.set_ylabel("Inference Time (ms)")
ax.set_xticks([0, 1])
ax.set_xticklabels(["NeuralNet", "MPC"])
ax.set_title("Inference Time Comparison")
ax.yaxis.grid(True)
ax.legend()
fig.tight_layout()
fig.savefig("inference_time_comparison.png")
plt.show()
