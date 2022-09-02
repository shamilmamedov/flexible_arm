#!/usr/bin/env python3

import numpy as np
from abc import ABC, abstractmethod

from flexible_arm_3dof import SymbolicFlexibleArm3DOF
from simulation import symbolic_RK4


class BaseEstimator(ABC):
    """ Abstract base class for estimators.
    All the estimators should implement estimate method
    """

    @abstractmethod
    def estimate(self, u, y):
        """
        :parameter y: measured output of the system
        """
        ...


class ExtendedKalmanFilter(BaseEstimator):
    """ Implements extended kalman filter
    """

    def __init__(self, model: "SymbolicFlexibleArm3DOF", x0_hat, P0, Q, R, ts) -> None:
        """
        :parameter model: symbolic model of the arm
        :parameter x0_hat: initial state estimate
        :parameter P0: initial estimate covariance
        :parameter Q: state covariance matrix
        :parameter R: measurement covariance matrix
        :parameter ts: sampling time
        """
        self.model = model
        self.P = P0
        self.Q = Q
        self.R = R
        self.x_hat = x0_hat
        self.ts = ts

        # Create a symbolic integrator for dynamics
        self.F = symbolic_RK4(self.model.x, self.model.u, self.model.ode, n=1)

        # Linearize the output equation
        self.C = np.zeros((self.model.ny, self.model.nx))
        self.A = np.zeros((self.model.nx, self.model.nx))

    def linearize(self, u):
        """ Linearizes dynamics
        """
        self.A = np.array(self.model.dF_dx(self.x_hat, u, self.ts))
        self.C = np.array(self.model.dh_dx(self.x_hat))

    def compute_Kalman_gain(self):
        t_ = self.R + self.C @ self.P @ self.C.T
        L = self.P @ self.C.T @ np.linalg.pinv(t_)
        return L

    def update(self, y):
        L = self.compute_Kalman_gain()
        self.x_hat = self.x_hat + L @ (y - np.array(self.model.h(self.x_hat)))
        self.P = self.P - L @ self.C @ self.P
        return self.x_hat

    def predict(self, u):
        self.linearize(u)
        self.P = self.A @ self.P @ self.A.T + self.Q
        self.x_hat = self.F(self.x_hat, u, self.ts)

    def estimate(self, u, y):
        self.predict(u)
        return self.update(y)
