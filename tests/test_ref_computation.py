import time
import numpy as np


from utils.gym_utils import (
    create_unified_flexiblearmenv_and_controller_and_safety_filter,
)
from envs.flexible_arm_3dof import (
    SymbolicFlexibleArm3DOF,
    get_rest_configuration,
    inverse_kinematics_rb,
    analytical_inverse_kinematics_rb,
    compute_reference_state_and_input
)


def main():
    env = create_unified_flexiblearmenv_and_controller_and_safety_filter(
        create_controller=False, add_wall_obstacle=True, create_safety_filter=False
    )[0]


    for _ in range(25):
        env.reset()
        env.render()
        time.sleep(2)

        pee_goal = env.xee_final
        qa = env._state[:env.model_sym.nq]

        x_ref, u_ref = compute_reference_state_and_input(env.model_sym, qa, pee_goal)
        env._state = x_ref.flatten()

        env.render()
        time.sleep(2)


if __name__ == "__main__":
    main()