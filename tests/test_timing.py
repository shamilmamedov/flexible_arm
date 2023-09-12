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
    SafetyWrapper,
)
import matplotlib.pyplot as plt

device = torch.device("cuda")
(
    env,
    controller,
    safety_filter,
) = create_unified_flexiblearmenv_and_controller_and_safety_filter(
    create_controller=True, add_wall_obstacle=True, create_safety_filter=True
)
dummy_observations, _ = env.reset()
dummy_observations = torch.from_numpy(dummy_observations.reshape(1, -1)).to(device)
reps = 1000

# --- SAC (No Safety Filter)---
agent = SAC(policy=SACPolicy, env=env, verbose=0, seed=0)
policy = agent.policy.to(device)

starter = torch.cuda.Event(enable_timing=True)
ender = torch.cuda.Event(enable_timing=True)
timings = np.zeros((reps, 1))
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
sac_mean = np.mean(timings)
sac_std = np.std(timings)
print(f"SAC Mean inference time: {sac_mean} ms")
print(f"SAC Std inference time: {sac_std} ms")

# --- DAGGER (No Safety Filter)---
policy = torch.load("trained_models/IL/DAGGER/2023-09-04_14-44/SEED_0/best_model.pt")
policy = policy.to(device)

starter = torch.cuda.Event(enable_timing=True)
ender = torch.cuda.Event(enable_timing=True)
timings = np.zeros((reps, 1))
# warmup
for _ in range(100):
    _ = policy(dummy_observations)
# timing
with torch.no_grad():
    for i in range(reps):
        starter.record()
        _ = policy._predict(dummy_observations)
        ender.record()
        # synchronize
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[i] = curr_time
dagger_mean = np.mean(timings)
dagger_std = np.std(timings)
print(f"DAGGER Mean inference time: {dagger_mean} ms")
print(f"DAGGER Std inference time: {dagger_std} ms")

# --- SAC (With Safety Filter)---
agent = SAC(policy=SACPolicy, env=env, verbose=0, seed=0)
policy = agent.policy.to(device)
safe_policy = SafetyWrapper(policy, safety_filter=safety_filter)


starter = torch.cuda.Event(enable_timing=True)
ender = torch.cuda.Event(enable_timing=True)
timings = np.zeros((reps, 1))
# warmup
for _ in range(100):
    _ = policy(dummy_observations)
# timing
with torch.no_grad():
    for i in range(reps):
        starter.record()
        unsafe_action = policy(dummy_observations)
        ender.record()
        # synchronize
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[i] = curr_time
        # apply safety filter
        _ = safe_policy._apply_safety(unsafe_action, dummy_observations)

safety_timings = (
    np.array(safety_filter.debug_timings).reshape(-1, 1) * 1000
)  # convert to ms
total_time = timings + safety_timings
sac_safe_mean = np.mean(total_time)
sac_safe_std = np.std(total_time)
print(f"SAC+SF Mean inference time (safe): {sac_safe_mean} ms")
print(f"SAC+SF Std inference time (safe): {sac_safe_std} ms")

# --- DAGGER (With Safety Filter)---
policy = torch.load("trained_models/IL/DAGGER/2023-09-04_14-44/SEED_0/best_model.pt")
safe_policy = SafetyWrapper(policy, safety_filter=safety_filter)
safety_filter.reset()

starter = torch.cuda.Event(enable_timing=True)
ender = torch.cuda.Event(enable_timing=True)
timings = np.zeros((reps, 1))
# warmup
for _ in range(100):
    _ = policy(dummy_observations)
# timing
with torch.no_grad():
    for i in range(reps):
        starter.record()
        unsafe_action = policy._predict(dummy_observations, deterministic=True)
        ender.record()
        # synchronize
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[i] = curr_time
        # apply safety filter
        _ = safe_policy._apply_safety(unsafe_action, dummy_observations)

safety_timings = (
    np.array(safety_filter.debug_timings).reshape(-1, 1) * 1000
)  # convert to ms
total_time = timings + safety_timings
dagger_safe_mean = np.mean(total_time)
dagger_safe_std = np.std(total_time)
print(f"DAGGER+SF Mean inference time (safe): {dagger_safe_mean} ms")
print(f"DAGGER+SF Std inference time (safe): {dagger_safe_std} ms")

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
    sac_mean,
    yerr=sac_std,
    align="center",
    alpha=0.9,
    ecolor="black",
    capsize=14,
    label="NeuralNet",
    color="blueviolet",
    zorder=3,
)
ax.bar(
    1,
    sac_safe_mean,
    yerr=sac_safe_std,
    align="center",
    alpha=0.9,
    ecolor="black",
    capsize=14,
    label="NeuralNet + Safety Filter",
    color="blueviolet",
    zorder=3,
)
ax.bar(
    2,
    mpc_mean,
    yerr=mpc_std,
    align="center",
    alpha=0.9,
    ecolor="black",
    capsize=14,
    label="MPC",
    color="blueviolet",
    zorder=3,
)
ax.set_facecolor("ghostwhite")
ax.grid(color="white", linestyle="-", linewidth=1, zorder=0)

ax.set_ylabel("Inference Time (ms)", fontdict={"fontsize": 14})
ax.set_title("Inference Time Comparison", fontdict={"fontsize": 16})
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["NN", "NN+SF", "MPC"])
ax.xaxis.set_tick_params(labelsize=12)

fig.tight_layout()
fig.savefig("plots/inference_time_comparison.png")
fig.savefig("plots/inference_time_comparison.pdf")
plt.show()
