import numpy as np

from animation import Panda3dAnimator
from flexible_arm_3dof import (SymbolicFlexibleArm3DOF,
                                get_rest_configuration,
                                inverse_kinematics_rb)

# P2P task 3
QA_t0 = np.array([0., np.pi/30, -np.pi/30])
QA_ref = np.array([0., -7.33e-2, 9.38e-1])

n_seg = 10
q_t0 = get_rest_configuration(QA_t0, n_seg)
q_ref = get_rest_configuration(QA_ref, n_seg)

flexible_robot = SymbolicFlexibleArm3DOF(n_seg)
# Panda3dAnimator(flexible_robot.urdf_path, 10, np.tile(q_t0.T, [2,1])).play()
Panda3dAnimator(flexible_robot.urdf_path, 10, np.tile(q_ref.T, [2,1])).play()