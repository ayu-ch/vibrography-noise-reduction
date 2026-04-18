"""
Gym environment for RL-based camera stabilization.

Wraps the motion generators from synthetic_frame.py directly — no image
rendering, no ORB, no RANSAC.  RANSAC output is simulated by adding
noise to ground-truth sway, which is sufficient because the RL agent
never sees the image, only the RANSAC decomposition.

Observation (70 floats):
    [0:4]   current raw measurement  [tx, ty, θ, scale]  (normalized)
    [4]     inlier quality  (inliers / saturate, clipped to 1)
    [5]     homography valid flag  (0 or 1)
    [6:38]  past 8 raw measurements  (4 × 8 = 32)
    [38:70] past 8 applied corrections  (4 × 8 = 32)

Action (4 floats, continuous, tanh-squashed → rescaled):
    [tx, ty, θ, scale]  — the correction to apply
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class StabilizerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, domain_randomize=True, seed=None,
                 episode_length=500, fps=100, history_length=8):
        super().__init__()
        self.episode_length = episode_length
        self.fps = fps
        self.history_length = history_length
        self.domain_randomize = domain_randomize

        # Action: [tx, ty, θ, scale_offset]  all in [-1, 1] (rescaled later)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

        # Observation: 4 + 1 + 1 + 4*H + 4*H = 6 + 8*H
        obs_dim = 6 + 8 * self.history_length
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        # Action = small DELTA on top of the raw RANSAC measurement.
        # The agent doesn't output the full correction — it outputs a
        # refinement. This makes learning much easier because the raw
        # RANSAC is already ~90% correct; the agent just fixes the noise.
        self.delta_tx_max = 5.0     # max adjustment ±5px on top of RANSAC
        self.delta_ty_max = 3.0     # max adjustment ±3px
        self.delta_theta_max = 0.02 # max adjustment ±0.02°
        self.delta_scale_max = 0.001

        # Normalization
        self.norm_t = 50.0
        self.norm_r = 0.05
        self.inlier_saturate = 4000

        # Reward weights
        self.rotation_weight = 100.0
        self.smoothness_weight = 0.1
        self.scale_weight = 1000.0

        # RANSAC simulation defaults (from real video analysis)
        self.ransac_noise_sigma = 1.14
        self.ransac_inlier_mean = 2480
        self.ransac_inlier_std = 519
        self.dropout_rate = 0.002

        self._rng = np.random.default_rng(seed)

    # ── Motion generators (same as synthetic_frame.py) ────────────────────────

    def _generate_sway(self, t):
        bx = self._sway_amp_x * np.sin(2 * np.pi * self._sway_freq_x * t)
        by = self._sway_amp_y * np.cos(2 * np.pi * self._sway_freq_y * t)
        self._walk_x = self._walk_x * self._walk_ar + self._rng.normal(0, self._walk_sigma)
        self._walk_y = self._walk_y * self._walk_ar + self._rng.normal(0, self._walk_sigma)
        dx = np.clip(bx + self._walk_x, -20.0, 20.0)
        dy = np.clip(by + self._walk_y, -20.0, 20.0)
        return dx, dy

    def _generate_rotation(self, t):
        return self._rot_amp * np.sin(2 * np.pi * self._rot_freq * t)

    # ── RANSAC simulation ─────────────────────────────────────────────────────

    def _simulate_ransac(self, gt_tx, gt_ty, gt_theta):
        """Simulate noisy RANSAC measurement from ground truth.

        Noise model calibrated from real video analysis:
        - tx noise: light-tailed (kurtosis -0.95), use uniform-ish noise
        - ty noise: heavy-tailed (kurtosis 1.81), use t-distribution
        - Jitter NOT proportional to inliers (real data shows mid-range
          inliers have MORE jitter from keyframe transitions)
        """
        inliers = int(np.clip(
            self._rng.normal(self.ransac_inlier_mean, self.ransac_inlier_std),
            20, 5000))

        dropout = self._rng.random() < self._dropout_rate
        if dropout:
            return 0.0, 0.0, 0.0, 1.0, 0, False

        # tx: light-tailed noise (uniform + small Gaussian)
        noise_tx = self._ransac_noise_sigma * (
            0.7 * self._rng.uniform(-1, 1) + 0.3 * self._rng.normal(0, 1))

        # ty: heavy-tailed noise (t-distribution with df=4)
        noise_ty = self._ransac_noise_sigma * 0.3 * self._rng.standard_t(4)

        # H maps current→reference, so measurement ≈ -sway
        meas_tx = -gt_tx + noise_tx
        meas_ty = -gt_ty + noise_ty
        meas_theta = -gt_theta + self._rng.normal(0, 0.005)
        meas_scale = 1.0 + self._rng.normal(0, 0.0002)

        return meas_tx, meas_ty, meas_theta, meas_scale, inliers, True

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step = 0
        self._walk_x = 0.0
        self._walk_y = 0.0

        if self.domain_randomize:
            # Ranges calibrated from real video data
            self._sway_amp_x = self._rng.uniform(10.0, 50.0)
            self._sway_amp_y = self._rng.uniform(1.0, 8.0)
            self._sway_freq_x = self._rng.uniform(0.05, 0.20)
            self._sway_freq_y = self._rng.uniform(0.05, 0.15)
            self._walk_sigma = self._rng.uniform(1.0, 5.0)
            self._walk_ar = self._rng.uniform(0.85, 0.98)
            self._rot_amp = self._rng.uniform(0.005, 0.05)
            self._rot_freq = self._rng.uniform(0.4, 1.2)
            self._ransac_noise_sigma = self._rng.uniform(0.5, 2.0)
            self._dropout_rate = self._rng.uniform(0.0, 0.01)
        else:
            # Defaults matching the real ArUco video
            self._sway_amp_x = 24.0
            self._sway_amp_y = 2.5
            self._sway_freq_x = 0.10
            self._sway_freq_y = 0.10
            self._walk_sigma = 3.0
            self._walk_ar = 0.95
            self._rot_amp = 0.012
            self._rot_freq = 0.8
            self._ransac_noise_sigma = 1.14
            self._dropout_rate = 0.002

        # History buffers (ring buffer)
        H = self.history_length
        self._raw_history = np.zeros((H, 4), dtype=np.float32)
        self._act_history = np.zeros((H, 4), dtype=np.float32)
        self._hist_idx = 0
        self._prev_action = np.zeros(4, dtype=np.float32)

        # Generate first measurement so observation is valid
        t = 0.0
        self._gt_sway_x, self._gt_sway_y = self._generate_sway(t)
        self._gt_rot = self._generate_rotation(t)
        (self._meas_tx, self._meas_ty, self._meas_theta,
         self._meas_scale, self._inliers, self._valid) = \
            self._simulate_ransac(self._gt_sway_x, self._gt_sway_y, self._gt_rot)

        obs = self._build_obs()
        return obs, {}

    def step(self, action):
        # Action = delta on top of raw RANSAC measurement
        # Final correction = RANSAC + agent's delta
        delta_tx    = float(action[0]) * self.delta_tx_max
        delta_ty    = float(action[1]) * self.delta_ty_max
        delta_theta = float(action[2]) * self.delta_theta_max
        delta_scale = float(action[3]) * self.delta_scale_max

        act_tx    = self._meas_tx + delta_tx
        act_ty    = self._meas_ty + delta_ty
        act_theta = self._meas_theta + delta_theta
        act_scale = self._meas_scale + delta_scale

        # ── Reward ────────────────────────────────────────────────────────
        # Residual = gt_sway + applied correction (H maps current→ref)
        residual_tx = self._gt_sway_x + act_tx
        residual_ty = self._gt_sway_y + act_ty
        residual_rot = self._gt_rot - act_theta

        accuracy = -(residual_tx ** 2 + residual_ty ** 2
                     + self.rotation_weight * residual_rot ** 2)

        delta = np.array([act_tx - self._prev_action[0],
                          act_ty - self._prev_action[1]], dtype=np.float32)
        smoothness = -self.smoothness_weight * float(np.sum(delta ** 2))

        scale_pen = -self.scale_weight * (act_scale - 1.0) ** 2

        reward = float(accuracy + smoothness + scale_pen)

        # ── Update history ────────────────────────────────────────────────
        raw_vec = np.array([
            self._meas_tx / self.norm_t,
            self._meas_ty / self.norm_t,
            self._meas_theta / self.norm_r,
            self._meas_scale - 1.0
        ], dtype=np.float32)

        act_vec = np.array([
            act_tx / self.norm_t,
            act_ty / self.norm_t,
            act_theta / self.norm_r,
            act_scale - 1.0
        ], dtype=np.float32)

        idx = self._hist_idx % self.history_length
        self._raw_history[idx] = raw_vec
        self._act_history[idx] = act_vec
        self._hist_idx += 1

        self._prev_action = np.array([act_tx, act_ty, act_theta, act_scale],
                                     dtype=np.float32)

        # ── Advance to next frame ─────────────────────────────────────────
        self._step += 1
        t = self._step / self.fps
        self._gt_sway_x, self._gt_sway_y = self._generate_sway(t)
        self._gt_rot = self._generate_rotation(t)
        (self._meas_tx, self._meas_ty, self._meas_theta,
         self._meas_scale, self._inliers, self._valid) = \
            self._simulate_ransac(self._gt_sway_x, self._gt_sway_y, self._gt_rot)

        obs = self._build_obs()
        terminated = self._step >= self.episode_length
        truncated = False

        return obs, reward, terminated, truncated, {}

    def _build_obs(self):
        """Build the 70-float observation vector."""
        # Current measurement (normalized)
        current = np.array([
            self._meas_tx / self.norm_t,
            self._meas_ty / self.norm_t,
            self._meas_theta / self.norm_r,
            self._meas_scale - 1.0,
        ], dtype=np.float32)

        quality = np.float32(
            min(self._inliers / self.inlier_saturate, 1.0))
        valid = np.float32(1.0 if self._valid else 0.0)

        # Ordered history: oldest first
        H = self.history_length
        idx = self._hist_idx % H
        order = [(idx + i) % H for i in range(H)]
        raw_flat = self._raw_history[order].flatten()
        act_flat = self._act_history[order].flatten()

        obs = np.concatenate([
            current,
            [quality, valid],
            raw_flat,
            act_flat,
        ]).astype(np.float32)

        return obs
