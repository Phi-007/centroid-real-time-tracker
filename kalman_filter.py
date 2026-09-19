import numpy as np
from typing import Tuple


class KalmanFilter:
    def __init__(self, x: float, y: float, dt: float = 1.0,
                 process_var: float = 1.0, meas_var: float = 10.0):
        self.dt = dt
        self.x = np.array([[x], [y], [0.0], [0.0]], dtype=float) # State vector [x, y, vx, vy]

        self.F = np.array([                                      # State transition
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0 ],
            [0, 0, 0, 1 ],
        ], dtype=float)

        self.H = np.array([                                      # Observation matrix: we observe x, y
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        q = process_var
        self.Q = q * np.eye(4)                                   # Process noise covariance Q (small process noise on position and velocity)

        r = meas_var
        self.R = r * np.eye(2)                                   # Measurement noise covariance R

        self.P = np.eye(4) * 500.0                               # Covariance matrix P with large initial uncertainty

    def predict(self):
        self.x = self.F.dot(self.x)                              # x = F x
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q       # P = F P F^T + Q
        return self.get_position()

    def update(self, meas_x: float, meas_y: float):
        z = np.array([[meas_x], [meas_y]], dtype=float)
        y = z - (self.H.dot(self.x))  # innovation
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        self.x = self.x + K.dot(y)
        I = np.eye(self.P.shape[0])
        self.P = (I - K.dot(self.H)).dot(self.P)
        return self.get_position()

    def get_position(self) -> Tuple[float, float]:
        return float(self.x[0, 0]), float(self.x[1, 0])