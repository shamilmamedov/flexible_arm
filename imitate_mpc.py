from copy import deepcopy
import numpy as np
import torch
from flexible_arm_3dof import FlexibleArm3DOF, SymbolicFlexibleArm3DOF
from gym_env import FlexibleArmEnv
from mpc_3dof import Mpc3dofOptions, Mpc3Dof
from utils import ControlMode

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

dt = 0.05
env = FlexibleArmEnv(n_seg=n_seg, dt=dt, q0=q0[:, 0], xee_final=x_ee_ref)

from stable_baselines3.common import policies


# create callable controller function as expert
class CallableExpert(policies.BasePolicy):
    def __init__(self, controller, observation_space, action_space):
        super().__init__(observation_space, action_space)
        self.controller = controller
        self.observation_space = observation_space
        self.action_space = action_space

    def _predict(self, observation, deterministic: bool = False):
        n_q = int(observation.tolist()[0].__len__() / 2)
        torques = self.controller.compute_torques(np.array([observation.tolist()[0][:n_q]]).transpose(),
                                                  np.array([observation.tolist()[0][n_q:]]).transpose())
        torques = torch.tensor(torques)
        return torques

    def __call__(self, observation):
        return self._predict(observation)

callable_expert = CallableExpert(controller, observation_space=env.observation_space, action_space=env.action_space)

# check environment
from stable_baselines3.common.env_checker import check_env
check_env(env)

# peform behavior cloning
import tempfile
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer

venv = DummyVecEnv([lambda: FlexibleArmEnv(n_seg=n_seg, dt=dt, q0=q0, xee_final=x_ee_ref)])

bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space
)

with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
    print(tmpdir)
    dagger_trainer = SimpleDAggerTrainer(
        venv=venv,
        scratch_dir=tmpdir,
        expert_policy=callable_expert,
        bc_trainer=bc_trainer,
    )
    dagger_trainer.train(10000, rollout_round_min_episodes=20)

# Evaluate and save trained policy
from stable_baselines3.common.evaluation import evaluate_policy
reward, _ = evaluate_policy(dagger_trainer.policy, env, 10)
print("Final reward: {}".format(reward))

# Save policy
dagger_trainer.policy.save("bc_policy_2")


