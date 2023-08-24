import numpy as np

from envs.flexible_arm_3dof import SymbolicFlexibleArm3DOF
from envs.gym_env import (
    FlexibleArmEnv,
    FlexibleArmEnvOptions,
    WallObstacle
    )
from utils.utils import StateType


SEED = 0
rng = np.random.default_rng(SEED)


def _create_env():
    # --- Create FlexibleArm environment ---
    n_seg = 5
    n_seg_mpc = 3

    # Env options
    R_Q = [3e-6] * 3
    R_DQ = [2e-3] * 3
    R_PEE = [1e-4] * 3
    env_options = FlexibleArmEnvOptions(
        n_seg=n_seg,
        n_seg_estimator=n_seg_mpc,
        sim_time=1.3,
        dt=0.01,
        qa_range_start=np.array([-np.pi/2, 0., -np.pi+0.05]),
        qa_range_end=np.array([3*np.pi/2, np.pi, np.pi-0.05]),
        contr_input_states=StateType.ESTIMATED,  # "real" if the n_seg is the same for the data and control env
        sim_noise_R=np.diag([*R_Q, *R_DQ, *R_PEE]),
        render_mode="human",
    )
    
    # Wall obstacle
    w = np.array([0., 1., 0.])
    b = np.array([0., -0.15, 0.5])
    wall = WallObstacle(w, b)

    # Create environment
    env = FlexibleArmEnv(env_options, obstacle=wall)
    return env

def test_obstacle():
    # Wall obstacle
    w = np.array([0., 1., 0.])
    b = np.array([0., -0.15, 0.5])
    wall = WallObstacle(w, b)

    np.testing.assert_array_equal(wall.w, w)
    np.testing.assert_array_equal(wall.b, b)

def test_obstacle_observation():
    env = _create_env()
    obs = env.reset()[0]
    w = obs[-6:-3]
    b = obs[-3:]

    np.testing.assert_array_equal(w, env.obstacle.w)
    np.testing.assert_array_equal(b, env.obstacle.b)


if __name__ == "__main__":
    test_obstacle()
    test_obstacle_observation()