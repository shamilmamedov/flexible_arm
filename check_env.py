from copy import deepcopy
import numpy as np
from stable_baselines3.ppo import MlpPolicy
from flexible_arm_3dof import FlexibleArm3DOF, SymbolicFlexibleArm3DOF
from gym_env import FlexibleArmEnv
from mpc_3dof import Mpc3dofOptions, Mpc3Dof
from utils import ControlMode
from stable_baselines3.common import policies

control_mode = ControlMode.SET_POINT
n_seg = 3
fa_ld = FlexibleArm3DOF(n_seg)
fa_sym_ld = SymbolicFlexibleArm3DOF(n_seg)

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

dq0 = np.zeros_like(q0)
x0 = np.vstack((q0, dq0))

dq_ref = np.zeros_like(q_ref)
x_ref = np.vstack((q_ref, dq_ref))
_, x_ee_ref = fa_ld.fk_ee(q_ref)
_, xee0 = fa_ld.fk_ee(q0)

# create expert MPC
mpc_options = Mpc3dofOptions(n_links=n_seg)
mpc_options.n = 30
mpc_options.tf = 1
mpc_options.r_diag *= 10
controller = Mpc3Dof(model=fa_sym_ld, x0=x0, x0_ee=xee0, options=mpc_options)
u_ref = np.zeros((fa_sym_ld.nu, 1))  # u_ref could be changed to some known value along the trajectory

# Choose one out of two control modes. Reference tracking uses a spline planner.
controller.set_reference_point(p_ee_ref=x_ee_ref, x_ref=x_ref, u_ref=u_ref)

env = FlexibleArmEnv(n_seg=n_seg, dt=0.05, q0=q0[:, 0], xee_final=x_ee_ref)


class CallableExpert(policies.BasePolicy):
    """
    Slightly modified version of policy regarding np and torch arrays.
    Todo. Merge with other expert
    """

    def __init__(self, controller, observation_space, action_space):
        super().__init__(observation_space, action_space)
        self.controller = controller
        self.observation_space = observation_space
        self.action_space = action_space

    def _predict(self, observation, deterministic: bool = False):
        nq = int(observation.__len__() / 2)
        torques = self.controller.compute_torques(np.expand_dims(observation[:nq], axis=1),
                                                  np.expand_dims(observation[nq:], axis=1))
        return torques

    def __call__(self, observation):
        return self._predict(observation)


# create MPC expert
expert = CallableExpert(controller, observation_space=env.observation_space, action_space=env.action_space)

# load trained policy
learned_policy = MlpPolicy(observation_space=env.observation_space, action_space=env.action_space,
                           lr_schedule=lambda x: 1)
learned_policy = learned_policy.load("bc_policy_1")

# run tests
n_eps = 3
use_trained = True
for i in range(n_eps):
    sum_reward = 0
    print("Episode {}".format(i))
    done = False
    obs = env.reset()
    while not done:
        action_exp = expert(obs)
        action_bc = learned_policy.predict(obs, deterministic=True)[0]
        print("Action exp: {}, Action bc: {}".format(action_exp, action_bc))
        if use_trained:
            action = action_bc
        else:
            action = action_exp
        (obs, reward, done, info) = env.step(action)
        sum_reward += reward
        if done:
            print("N steps: {}".format(env.no_intg_steps))
    print("Sum of rewards: {}".format(sum_reward))
