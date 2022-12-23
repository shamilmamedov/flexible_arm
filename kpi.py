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
        pee[k, :] = np.array(model.p_ee(qk)).flatten()

    out = 0.
    for k in range(ns - 1):
        out += np.linalg.norm(pee[k + 1, :] - pee[k, :])

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
        dn = ns - n_first_entry
        # print(f"The EE stayed inside the ball for {dn} samples")
        return n_first_entry
    else:
        raise NotImplementedError


def constraint_violation(q: np.ndarray, dq: np.ndarray, u: np.ndarray,
                         model: SymbolicFlexibleArm3DOF, 
                         wall_params: dict = None):
    ns_q = q.shape[0]
    u_constr_violated = np.full_like(u, False)
    dqa_constr_violated = np.full((ns_q, 3), False)

    # Input constraint violation 
    for k, uk in enumerate(u):
        u_constr_violated[k,:] = np.logical_or(uk > model.tau_max, 
                                               uk < -model.tau_max) 

    # Joint velocity constraint violation
    for k, dqak in enumerate(dq[:,model.qa_idx]) :
        dqa_constr_violated[k,:] = np.logical_or(dqak > model.dqa_max,
                                                 dqak < -model.dqa_max)

    # Wall constraint violation
    if wall_params is not None:
        try:
            wa_ = wall_params['wall_axis']
            wv_ = wall_params['wall_value']
            wall_constr_violated = np.full((ns_q,), False)
            for k, qk in enumerate(q):
                pee_k = np.array(model.p_ee(qk))
                if wall_params['wall_pos_side']:
                    wall_constr_violated[k] = pee_k[wa_] > wv_
                else:
                    wall_constr_violated[k] = pee_k[wa_] < wv_
        except KeyError:
            print("Some of the wall parameters are missing!")
    else:
        wall_constr_violated = None

    return (np.sum(u_constr_violated, axis=0), np.sum(dqa_constr_violated, axis=0),
            np.sum(wall_constr_violated))
