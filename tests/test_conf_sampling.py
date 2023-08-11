import numpy as np


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
        qa_range_end=np.array([1.0, 1.0, 1.0]),
        contr_input_states=StateType.ESTIMATED,  # "real" if the n_seg is the same for the data and control env
        sim_noise_R=np.diag([*R_Q, *R_DQ, *R_PEE]),
        render_mode="human",
    )
    env = FlexibleArmEnv(env_options)
    

def main(n_seg: int = 3):
    robot = SymbolicFlexibleArm3DOF(n_seg)

    qa = np.array([0., np.pi/4, -np.pi/4])
    q = get_rest_configuration(qa, n_seg)
    p_ee = np.array(robot.p_ee(q)) 
    p_goal = p_ee + np.random.uniform(-0.1, 0.1, size=(3,1))
    
    renderer = Panda3dRenderer(robot.urdf_path)
    renderer.draw_sphere(np.array([1., 0., 1.]))
    for _ in range(100):
        renderer.render(q)

    # viz = FlexibleArmVisualizer(robot.urdf_path, dt=0.01)
    # viz.visualize_configuration(q, p_goal)
    # viz.visualize_configuration(q)

    # q = q.repeat(100, axis=1).T
    # viz.visualize_trajectory(q, p_goal)
    # viz.visualize_trajectory(q)


if __name__ == '__main__':
    main(n_seg=5)