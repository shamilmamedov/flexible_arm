#!/usr/bin/env python3

import numpy as np
from abc import ABC, abstractmethod


class BaseController(ABC):
    """ Abstract base class for controllers
    All the controllers must implement compute_torques 
    attrubute
    """

    @abstractmethod
    def compute_torques(self, q, dq):
        ...


class DummyController(BaseController):
    """ Controller that always returns zero torque
    """

    def __init__(self, n_joints=1) -> None:
        self.n_joints = n_joints

    def compute_torques(self, q, dq):
        return np.zeros((self.n_joints, 1))


class ConstantController(BaseController):
    """ Controller that always returns constant torque
    """

    def __init__(self, n_joints=1, constant=0) -> None:
        self.n_joints = n_joints
        self.constant = constant

    def compute_torques(self, q, dq):
        output = np.zeros((self.n_joints, 1))
        output[0] = self.constant
        return output


class PDController(BaseController):
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
        return self.Kp * (self.q_ref - q[0]) - self.Kd * dq[0]
