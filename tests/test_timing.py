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
env, controller, _ = create_unified_flexiblearmenv_and_controller_and_safety_filter(
    create_controller=True, add_wall_obstacle=True, create_safety_filter=False
)
agent = SAC(policy=SACPolicy, env=env, verbose=0, seed=0)
policy = agent.policy.to(device)
dummy_observations, _ = env.reset()
dummy_observations = torch.from_numpy(dummy_observations.reshape(1, -1)).to(device)


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
for _ in range(reps):
    controller.predict(dummy_observations.cpu())
mpc_mean, mpc_std, _, _ = controller.controller.get_timing_statistics()
mpc_mean *= 1000  # convert to ms
mpc_std *= 1000  # convert to ms
print(f"Mean MPC time: {mpc_mean} ms")
print(f"Std MPC time: {mpc_std} ms")


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
    mpc_mean,
    yerr=mpc_std,
    align="center",
    alpha=0.5,
    ecolor="black",
    capsize=10,
    label="MPC",
)
ax.set_yscale("log")
ax.set_ylabel("Log Inference Time (ms)")
ax.set_xticks([0, 1])
ax.set_xticklabels(["NeuralNet", "MPC"])
ax.set_title("Inference Time Comparison (Log Scale)")
ax.yaxis.grid(True)
ax.legend()
fig.tight_layout()
fig.savefig("inference_time_comparison.png")
plt.show()
