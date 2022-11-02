import numpy as np

from flexible_arm_3dof import get_rest_configuration, FlexibleArm3DOF, SymbolicFlexibleArm3DOF
from imitator import ImitatorOptions, Imitator
from mpc_3dof import Mpc3dofOptions, Mpc3Dof

if __name__ == "__main__":
    imitator_options = ImitatorOptions(dt=0.01)
    imitator_options.environment_options.sim_time = 4
    imitator_options.environment_options.n_seg = 3

    # Create FlexibleArm instances
    n_seg_mpc = 3
    fa_sym_ld = SymbolicFlexibleArm3DOF(n_seg_mpc)
    qa0 = np.array([0.1, 1.5, 0.5])
    q0 = get_rest_configuration(qa0, n_seg_mpc)
    dq0 = np.zeros_like(q0)
    x0 = np.vstack((q0, dq0))
    fa_ld = FlexibleArm3DOF(n_seg_mpc)
    _, x_ee_ref = fa_ld.fk_ee(q0)

    # Create mpc options and controller
    mpc_options = Mpc3dofOptions(n_seg=n_seg_mpc, tf=0.3)
    mpc_options.n = 30
    controller = Mpc3Dof(model=fa_sym_ld, x0=x0, x0_ee=x_ee_ref, options=mpc_options)

    # Create estimator
    estimator = None

    # create imitator and train
    imitator = Imitator(options=imitator_options, expert_controller=controller, estimator=estimator)
    imitator.train()
