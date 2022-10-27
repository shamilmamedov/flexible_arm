import numpy as np
from abc import ABC, abstractmethod

from flexible_arm_3dof import SymbolicFlexibleArm3DOF


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

    def __init__(self, model: "SymbolicFlexibleArm3DOF", x0_hat, P0, Q, R) -> None:
        """
        :parameter model: symbolic model of the arm
        :parameter x0_hat: initial state estimate
        :parameter P0: initial estimate covariance
        :parameter Q: state covariance matrix
        :parameter R: measurement covariance matrix
        """
        self.model = model
        self.P = P0
        self.Q = Q
        self.R = R
        self.x_hat = x0_hat

    def compute_Kalman_gain(self, C):
        """ Computes Kalman Gain given C
        """
        t_ = self.R + C @ self.P @ C.T
        L = self.P @ C.T @ np.linalg.pinv(t_)
        return L

    def update(self, y):
        """ Implements correction equations of the EKF
        """
        # Linearize measurement equation
        C = np.array(self.model.dh_dx(self.x_hat))

        # Compute Kalman gain
        L = self.compute_Kalman_gain(C)

        # Correct state predictions
        self.x_hat = self.x_hat + L @ (y - np.array(self.model.h(self.x_hat)))

        # Correct covariance predictions
        self.P = self.P - L @ C @ self.P
        return self.x_hat

    def predict(self, u):
        """ Implements prediction equations of the EKF
        """
        # Linearize state transition equations
        A = np.array(self.model.dF_dx(self.x_hat, u))
        
        # Predict state covraince
        self.P = A @ self.P @ A.T + self.Q

        # Predict next state
        self.x_hat = np.array(self.model.F(self.x_hat, u))

    def estimate(self, y, u=None):
        """ Combines estimation and prediction

        In the iteration when there is no control input available
        we can only update the state estimate
        """
        if u is None:
            return self.update(y)
        else:
            self.predict(u)
            return self.update(y)
