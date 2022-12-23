import numpy as np
from flexible_arm_3dof import get_rest_configuration

N_SEG = 10
FILE_PATH = 'flexible_arm_ws/src/fa_setup/js_traj.csv'

# Active joints positions
# qa = np.array([[0., np.pi/4, -np.pi/4]])
# qa = np.array([0., np.pi / 10, -np.pi / 8])
# qa = np.array([0., np.pi/20, -np.pi/20])
# qa = np.array([-np.pi/4, np.pi/30, -np.pi/30])

# Data taken from imitation builder
delta_wall_angle = np.pi/40
## qa = np.array([-np.pi/2 - delta_wall_angle, np.pi/7, -np.pi/5])
qa = np.array([-np.pi/2 - delta_wall_angle, np.pi / 10, np.pi / 8])
## qa = np.array([-np.pi/4 - delta_wall_angle, np.pi/20, np.pi/4])
# qa = np.array([-delta_wall_angle, np.pi / 7, -np.pi / 5])


# Get rest configuration
q = get_rest_configuration(qa, N_SEG).T

# Create reference by padding q
q_ref = np.pad(q, ((300,300), (0,0)), mode='edge')

# Save the configuration in a csv file for RViz
np.savetxt(FILE_PATH, q_ref, delimiter=',')