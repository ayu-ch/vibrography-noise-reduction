"""
Python port of MotionKalman from ransac_stabilizer.cpp (lines 71-159).

4-state position model: [tx, ty, θ, scale], F=I, adaptive R.
Used for apples-to-apples comparison against the RL policy on the
same simulated RANSAC measurement streams.
"""

import numpy as np


class MotionKalman:
    """Mirrors the C++ MotionKalman exactly."""

    KF_INLIER_SATURATE = 400

    def __init__(self):
        # State [tx, ty, θ, scale]
        self.x = np.zeros(4, dtype=np.float64)  # state post
        self.P = np.eye(4, dtype=np.float64)     # error cov post

        # F = I (position model)
        self.F = np.eye(4, dtype=np.float64)

        # H = I (direct observation)
        self.H = np.eye(4, dtype=np.float64)

        # Q: process noise (inter-frame motion variance)
        self.Q = np.diag([4.0, 4.0, 6e-4, 1e-5])

        # R_base: measurement noise (scaled adaptively)
        self.R_base = np.diag([2.0, 2.0, 2.5e-3, 1e-4])

        self.initialized = False

    def update(self, tx_m, ty_m, theta_m, scale_m, inliers):
        """Predict + correct with RANSAC measurement."""
        z = np.array([tx_m, ty_m, theta_m, scale_m], dtype=np.float64)

        if not self.initialized:
            self.x = z.copy()
            self.initialized = True
            return self.x.copy()

        # Predict (F=I)
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Adaptive R
        quality = min(1.0, inliers / self.KF_INLIER_SATURATE)
        quality = max(quality, 0.05)
        R = self.R_base / quality

        # Correct
        S = self.H @ P_pred @ self.H.T + R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ (z - self.H @ x_pred)
        self.P = (np.eye(4) - K @ self.H) @ P_pred

        return self.x.copy()

    def predict_only(self):
        """No measurement — advance with motion model only."""
        if not self.initialized:
            return np.array([0.0, 0.0, 0.0, 1.0])

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()
