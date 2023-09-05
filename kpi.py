import numpy as np
from typing import Union, List

from envs.flexible_arm_3dof import SymbolicFlexibleArm3DOF


def _parse_observation(obs: np.ndarray, n_seg: int = 3) -> np.ndarray:
    nq = 3 + n_seg * 2
    nx = 2 * nq
    q_idxs = np.arange(0, nq)
    dq_idxs = np.arange(nq, nx)
    pee_idxs = np.arange(nx, nx+3)
    pee_goal_idxs = np.arange(nx+3, nx+6)
    
    q = obs[:, q_idxs]
    dq = obs[:, dq_idxs]
    pee = obs[:, pee_idxs]
    pee_goal = obs[:, pee_goal_idxs]
    return q, dq, pee, pee_goal


def find_index_after_true(binary_array):
    for i, value in enumerate(binary_array):
        if value:
            if all(binary_array[i+1:]):
                return i
    return None


def _time2reach_goal(traj, r: float, n_seg: int = 3) -> Union[int, None]:
    """ Computes amount of time that was required to
    arrive to a ball of specified radius and stay there
    
    :parameter r: a ball of radius around the reference

    :return: a sample at which the robot entered the ball
             of radius r and never left afterwards
    """
    q, dq, pee, pee_goal = _parse_observation(traj.obs, n_seg)
    traj_len = q.shape[0]

    # a boolean vector indicating if the ee is inside
    # the ball of radius r
    inside = np.full((traj_len,), False)

    n_first_entry = None
    for k, (pee_k, pee_goal_k) in enumerate(zip(pee, pee_goal)):
        if np.linalg.norm(pee_goal_k - pee_k) <= r:
            if n_first_entry == None:
                n_first_entry = k
            inside[k] = True

    if n_first_entry is None:
        return None
    
    idx = find_index_after_true(inside[n_first_entry:])
    if idx is not None:
        return n_first_entry + idx
    else:
        return idx


def time2reach_goal(trajs, r: float, n_seg: int = 3) -> List[int]:
    return [_time2reach_goal(traj, r, n_seg) for traj in trajs]


def _path_length(traj, n_seg: int = 3) -> float:
    """ Computes path length of the end-effector based on
    linear approximation: between two sampling times the
    path is assumed to be linear and the length in a sense
    of Euclidean distance is computed

    :return: the path (arc) length of the end-effector
    """

    q, dq, pee, pee_goal = _parse_observation(traj.obs, n_seg)
    traj_len = q.shape[0]

    out = 0.
    for k in range(traj_len - 1):
        out += np.linalg.norm(pee[k + 1, :] - pee[k, :])
    return out


def path_length(trajs) -> List[float]:
    return [_path_length(traj) for traj in trajs]


def _input_constraint_violation(u, input_range: tuple) -> np.ndarray:
    """ Computes the number of times the input constraints
    were violated during the trajectory

    :return: a vector of length 3 indicating the number of
             times each of the input constraints was violated
    """
    u_min, u_max = input_range

    u_constr_violated = np.full_like(u, False)

    for k, uk in enumerate(u):
        u_constr_violated[k,:] = np.logical_or(uk > u_max, uk < u_min) 
        
    return np.sum(u_constr_violated, axis=0)


def _joint_velocity_constraint_violation(dqa, dqa_range: float) -> np.ndarray:
    """ Computes the number of times the joint velocity
    constraints were violated during the trajectory

    :return: a vector of length 3 indicating the number of
             times each of the joint velocity constraints was violated
    """
    dqa_min, dqa_max = dqa_range

    dqa_constr_violated = np.full_like(dqa, False)
    for k, dqak in enumerate(dqa) :
        dqa_constr_violated[k,:] = np.logical_or(dqak > dqa_max,
                                                 dqak < dqa_min)
        if k > 50 and k < 60:
            print(dqak)
            print(dqa_max)
            print(dqa_min)
    return np.sum(dqa_constr_violated, axis=0)
        

def _wall_constraint_violation(q, wall_params: dict) -> np.ndarray:
    pass


def _constraint_violation(traj, n_seg: int = 3):
    # Parse trajectory
    u = traj.acts
    q, dq, pee, pee_goal = _parse_observation(traj.obs, n_seg)

    # Instantiate a model
    robot = SymbolicFlexibleArm3DOF(n_seg)
    u_range = (-robot.tau_max, robot.tau_max)
    dqa_range = (-robot.dqa_max, robot.dqa_max)
    dqa = dq[:, robot.qa_idx]

    # Compute constraint violation
    u_constr_violated = _input_constraint_violation(u, u_range)
    dqa_constr_violated = _joint_velocity_constraint_violation(dqa, dqa_range)
    return u_constr_violated, dqa_constr_violated



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
