import time
import numpy as np


from utils.gym_utils import (
    create_unified_flexiblearmenv_and_controller_and_safety_filter,
)
from envs.flexible_arm_3dof import (
    SymbolicFlexibleArm3DOF,
    get_rest_configuration,
    inverse_kinematics_rb,
)


def compute_reference_state_and_input(robot, qa: np.ndarray, p_ee_ref: np.ndarray):
    """
    Compute reference states genaralized coordinates (angles q, angular velocities dq)
    related to MPC model out of endeffetor states.

    @param robot: Robot model
    @param qa: current configuration of the active joints
    @param p_ee_ref: Endefector Cartesian reference position

    @return: reference states x = (q, dq) and input u
    """
    # Try to find a solution for the reference endeffector position
    qa = inverse_kinematics_rb(p_ee_ref, q_guess=qa).ravel()
    q = get_rest_configuration(qa, robot.n_seg).T

    # Compute stateic error due to flexiblity
    p_ee = np.array(robot.p_ee(q))
    delta_p_ee = p_ee_ref - p_ee

    # Try to find a solution for the augmented endeffector position
    if np.linalg.norm(delta_p_ee) > 2e-2:
        try:
            qa_aug = inverse_kinematics_rb(p_ee_ref + delta_p_ee, q_guess=qa).ravel()
            q = get_rest_configuration(qa_aug, robot.n_seg)
        except RuntimeError:
            pass

    dq = np.zeros_like(q)
    x_ref = np.vstack((q, dq))
    u_ref = robot.gravity_torque(q)
    return x_ref, u_ref


def main():
    env = create_unified_flexiblearmenv_and_controller_and_safety_filter(
        create_controller=False, add_wall_obstacle=True, create_safety_filter=False
    )[0]


    for _ in range(25):
        env.reset()
        env.render()
        time.sleep(2)

        pee_goal = env.xee_final
        qa = env._state[env.model_sym.qa_idx]

        x_ref, u_ref = compute_reference_state_and_input(env.model_sym, qa, pee_goal)
        env._state = x_ref.flatten()

        env.render()
        time.sleep(2)


if __name__ == "__main__":
    main()