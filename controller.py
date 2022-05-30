#!/usr/bin/env python3

import numpy as np

class BaseController():
    """ Base class for controllers
    All the controllers must implement compute_torques 
    attrubute
    """
    pass


class DummyController:
    """ Controller that always returns zero torque
    """
    def __init__(self, n_joints=1) -> None:
        self.n_joints = n_joints

    def compute_torques(self, q, dq):
        return np.zeros((self.n_joints,1))


class PDController:
    """ Proportional-Derivative controller
    """
    def __init__(self, Kp, Kd, q_ref) -> None:
        """
        :parameter Kp: proportional gain
        :parameter Kd: derivative gain
        :parmater q_ref: reference joint position
        """
        self.Kp = Kp
        self.Kd = Kd
        self.q_ref = q_ref

    def compute_torques(self, q, dq):
        return self.Kp*(self.q_ref - q) - self.Kd*dq