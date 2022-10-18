#!/usr/bin/env python3
from typing import List, Tuple

import numpy as np
from abc import ABC, abstractmethod


class BaseController(ABC):
    """ Abstract base class for controllers
    All the controllers must implement compute_torques 
    attrubute
    """

    @abstractmethod
    def compute_torques(self, q, dq, t=None):
        ...


class DummyController(BaseController):
    """ Controller that always returns zero torque
    """

    def __init__(self, n_joints=1) -> None:
        self.n_joints = n_joints

    def compute_torques(self, q, dq, t=None):
        return np.zeros((self.n_joints, 1)).T


class OfflineController(BaseController):
    """
    This controler variant just outputs predefined control signals. It can be used e.g. in a forwards simulation of the
    system
    """

    def __init__(self, n_joints=1) -> None:
        self.n_joints = n_joints
        self.u_pre = None
        self.iter = 0

    def set_u(self, u: np.ndarray):
        """
        Set the predefined control signals
        @param u: predefined controls with size (n_time_steps, n_controls)
        """
        assert len(u.shape) == 2
        assert u.shape[1] == 3
        self.u_pre = u
        self.iter = 0

    def compute_torques(self, q, dq, t=None) -> np.ndarray:
        if self.iter < self.u_pre.shape[0]:
            return self.u_pre[self.iter, :]
        else:
            return np.array([0., 0., 0.])


class ConstantController(BaseController):
    """ Controller that always returns a constant torque
    """

    def __init__(self, n_joints=1, constant=0) -> None:
        self.n_joints = n_joints
        self.constant = constant

    def compute_torques(self, q, dq, t=None):
        output = np.zeros((self.n_joints, 1))
        output[0] = self.constant
        return output


class FeedforwardController(BaseController):
    """ A controller that returns predefined sequence of torques
    """

    def __init__(self, u) -> None:
        """
        :parameter u: [ns x nj] sequence of inputs; ns is the
                      number of samples; nj is the number of joints
        """
        self.u = u
        self.ns = u.shape[0]
        self.iter = 0

    def compute_torques(self, q, dq, t=None):
        if self.iter < self.ns:
            tau = self.u[[self.iter], :]
            self.iter += 1
            return tau
        else:
            raise RuntimeError


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

    def compute_torques(self, q, dq, t=None):
        return self.Kp * (self.q_ref - q[0]) - self.Kd * dq[0]


class PDController3Dof(BaseController):
    """ Proportional-Derivative controller
    """

    def __init__(self, Kp: Tuple, Kd: Tuple, n_seg: int, q_ref: np.ndarray) -> None:
        """
        :parameter Kp: proportional gain
        :parameter Kd: derivative gain
        :parmater q_ref: reference joint position
        """
        self.Kp = Kp
        self.Kd = Kd
        self.q_ref = q_ref
        self.n_seg = n_seg

    def compute_torques(self, q, dq, t=None):
        out0 = self.Kp[0] * (self.q_ref[0] - q[0]) - self.Kd[0] * dq[0]
        out1 = self.Kp[1] * (self.q_ref[1] - q[1]) - self.Kd[1] * dq[1]
        out2 = self.Kp[2] * (self.q_ref[self.n_seg + 2] -
                             q[self.n_seg + 2]) - self.Kd[2] * dq[self.n_seg + 2]

        return np.array([out0, out1, out2]).transpose()
