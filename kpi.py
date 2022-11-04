import numpy as np
from typing import Union

from flexible_arm_3dof import SymbolicFlexibleArm3DOF


def path_length(q: np.ndarray, model: SymbolicFlexibleArm3DOF) -> float:
    """ Computes path length of the end-effector based on
    linear approximation: between two sampling times the
    path is assumed to be linear and the length in a sense
    of Euclidean distance is computed

    :parameter q: the robot configuration vector
    :parameter model: model of the robot (used in simulation)

    :return: the path (arc) length
    """
    # Compute the end-effector path
    ns = q.shape[0]
    pee = np.zeros((ns, 3))
    for k, qk in enumerate(q):
        pee[k,:] = np.array(model.p_ee(qk)).flatten()

    out = 0.
    for k in range(ns-1):
        out += np.linalg.norm(pee[k+1,:] - pee[k,:])

    return out


def execution_time(q: np.ndarray, model: SymbolicFlexibleArm3DOF,
                   pee_ref: np.ndarray, r: float) -> Union[int, None]:
    """ Computes amount of time that was required to
    arrive to a ball of specified radius and stay there
    
    :parameter q: the robot configuration vector
    :parameter model: model of the robot (used in simulation)
    :parameter pee_ref:  reference end-effector position
    :parameter r: a ball of radius around the reference

    :return: a sample at which the robot entered the ball
             of radius r and never left afterwards
    """
    ns = q.shape[0]

    # a boolean vector indicating if the ee is inside
    # the ball of radius r
    inside = np.full((ns,), False)
    
    n_first_entry = None
    for k, qk in enumerate(q):
        pee_k = np.array(model.p_ee(qk))
        if np.linalg.norm(pee_ref - pee_k) <= r:
            if n_first_entry == None:
                n_first_entry = k
            inside[k] = True

    if np.all(inside[n_first_entry:]):
        return n_first_entry
    else:
        return NotImplementedError