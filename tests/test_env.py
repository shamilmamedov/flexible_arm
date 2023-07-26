"""
RUN COMMAND: python -m tests.test_env

This is a minimal example for testing the FlexibleArmEnv.
"""

from copy import deepcopy
import numpy as np
from envs.flexible_arm_3dof import FlexibleArm3DOF
from envs.gym_env import FlexibleArmEnv, FlexibleArmEnvOptions

n_seg = 3
fa_ld = FlexibleArm3DOF(n_seg)

# Create initial state simulated system
angle0_0 = 0.
angle0_1 = 0
angle0_2 = np.pi / 20

q0 = np.zeros((fa_ld.nq, 1))
q0[0] += angle0_0
q0[1] += angle0_1
q0[1 + n_seg + 1] += angle0_2

# Compute reference
q_ref = deepcopy(q0)
q_ref[0] += np.pi / 2
q_ref[1] += 0
q_ref[1 + n_seg + 1] += -np.pi / 20

# NOTE: what are the ones below? should they be used in the env?
dq0 = np.zeros_like(q0)
x0 = np.vstack((q0, dq0))

dq_ref = np.zeros_like(q_ref)
x_ref = np.vstack((q_ref, dq_ref))
_, x_ee_ref = fa_ld.fk_ee(q_ref)
_, xee0 = fa_ld.fk_ee(q0)

# create environment
# Measurement noise covairance parameters
R_Q = [3e-6] * 3
R_DQ = [2e-3] * 3
R_PEE = [1e-4] * 3
env_options = FlexibleArmEnvOptions(n_seg=n_seg, dt=0.05, qa_start=q0[0:3, 0], qa_end=q_ref[0:3, 0], contr_input_states="real", sim_noise_R=np.diag([*R_Q, *R_DQ, *R_PEE])) # NOTE what should the inputs be here?
env = FlexibleArmEnv(env_options)

# run tests
n_eps = 3
use_trained = True
for i in range(n_eps):
    sum_reward = 0
    print("Episode {}".format(i))
    done = False
    obs = env.reset()
    while not done:
        action = env.action_space.sample()
        print("Random Action: {}".format(action))
        (obs, reward, done, info) = env.step(action)
        sum_reward += reward
        if done:
            print("N steps: {}".format(env.no_intg_steps))
    print("Sum of rewards: {}".format(sum_reward))
