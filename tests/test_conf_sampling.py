import numpy as np
import time

from envs.flexible_arm_3dof import SymbolicFlexibleArm3DOF, get_rest_configuration
from animation import Panda3dRenderer
from envs.gym_env import (
    FlexibleArmEnv,
    FlexibleArmEnvOptions,
)
from utils.utils import StateType


def _create_env():
    # --- Create FlexibleArm environment ---
    # --- data env ---
    n_seg_data = 5
    # --- control env ---
    n_seg_control = 3

    # Create initial state data env
    # base-rotation, base-bend, elbow-bend
    qa_initial = np.array([np.pi / 2, np.pi / 10, -np.pi / 8])
    qa_final = np.array([0.0, 2 * np.pi / 5, -np.pi / 3])

    qa_range_start = np.array([-2*np.pi, 0., -np.pi+0.05])
    qa_range_end = np.array([2*np.pi, np.pi, np.pi-0.05])

    # create data environment
    R_Q = [3e-6] * 3
    R_DQ = [2e-3] * 3
    R_PEE = [1e-4] * 3
    env_options = FlexibleArmEnvOptions(
        n_seg=n_seg_data,
        n_seg_estimator=n_seg_control,
        sim_time=1.3,
        dt=0.01,
        qa_start=qa_initial,
        qa_end=qa_final,
        qa_range_start=qa_range_start,
        qa_range_end=qa_range_end,
        contr_input_states=StateType.ESTIMATED,  # "real" if the n_seg is the same for the data and control env
        sim_noise_R=np.diag([*R_Q, *R_DQ, *R_PEE]),
        render_mode="human",
    )
    return FlexibleArmEnv(env_options)
    

def main():
    env = _create_env()
    for _ in range(10):
        env.reset()
        env.render()
        time.sleep(2)


if __name__ == '__main__':
    main()