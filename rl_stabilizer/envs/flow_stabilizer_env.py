"""
RL Environment — Phase Correlation + CNN Rotation Delta

Architecture:
    Frame t + Reference → phase correlation → base tx, ty (sub-pixel accurate)
                        → downsampled frame pair → CNN detects rotation pattern
                        → agent outputs delta [δtx, δty, δθ, δscale]
                        → final correction = phase_corr + delta

Why phase correlation instead of Farneback:
    - Farneback at 480×360 gives 28px residual (useless)
    - Phase correlation gives <0.5px residual at ANY resolution
    - 10x faster, no parameters

What the CNN learns:
    - Rotation residual (phase correlation only estimates translation)
    - Scale drift
    - Non-rigid distortions that phase correlation misses

Observation:
    [0:2]   phase correlation shift [dx, dy] normalized
    [2:4]   small reference patch (centre 32×32) flattened stats (mean, std)
    [4:6]   small current patch (centre 32×32) flattened stats
    [6:8]   prev_delta [δtx, δty]
    [8:10]  prev_delta [δθ, δscale]
    [10]    time normalized
    [11:11+flow_size] downsampled frame difference (captures rotation/distortion)

Action (4 floats in [-1, 1]):
    Delta on top of phase correlation: [δtx, δty, δθ, δscale]
    Rescaled to ±2px, ±2px, ±0.1°, ±0.002
"""

import gymnasium as gym
import numpy as np
import cv2
from gymnasium import spaces


class FlowStabilizerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, domain_randomize=True, seed=None,
                 frame_w=480, frame_h=360,
                 diff_w=60, diff_h=45,
                 episode_length=300, fps=100):
        super().__init__()
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.diff_w = diff_w
        self.diff_h = diff_h
        self.episode_length = episode_length
        self.fps = fps
        self.domain_randomize = domain_randomize

        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

        # Observation: phase_corr(2) + patch_stats(4) + prev_delta(4) + time(1)
        #            + frame_diff (diff_w × diff_h) flattened
        self.diff_size = diff_w * diff_h
        obs_dim = 2 + 4 + 4 + 1 + self.diff_size
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        # Scale factor: training resolution vs real video resolution
        self._motion_scale = frame_w / 1920.0

        # Delta scaling (small adjustments on top of phase correlation)
        scale = self._motion_scale
        self.delta_tx_max = 2.0 * scale
        self.delta_ty_max = 2.0 * scale
        self.delta_theta_max = 0.1    # rotation doesn't scale with resolution
        self.delta_scale_max = 0.002

        # Normalization
        self.tx_norm = 50.0 * scale
        self.ty_norm = 20.0 * scale

        # Reward weights
        self.rotation_weight = 100.0
        self.smoothness_weight = 0.05
        self.scale_weight = 1000.0

        self._rng = np.random.default_rng(seed)

    # ── Texture ───────────────────────────────────────────────────────────

    def _make_texture(self):
        tex = np.full((self.frame_h, self.frame_w), 200, dtype=np.uint8)
        n = int(self.frame_w * self.frame_h * 0.06)
        xs = self._rng.integers(0, self.frame_w, size=n)
        ys = self._rng.integers(0, self.frame_h, size=n)
        vals = self._rng.integers(0, 180, size=n).astype(np.uint8)
        tex[ys, xs] = vals
        tex = cv2.GaussianBlur(tex, (3, 3), 0.8)
        return tex

    # ── Motion ────────────────────────────────────────────────────────────

    def _generate_sway(self, t):
        bx = self._sway_amp_x * np.sin(2 * np.pi * self._sway_freq_x * t)
        by = self._sway_amp_y * np.cos(2 * np.pi * self._sway_freq_y * t)
        self._walk_x = self._walk_x * self._walk_ar + self._rng.normal(0, self._walk_sigma)
        self._walk_y = self._walk_y * self._walk_ar + self._rng.normal(0, self._walk_sigma * 0.3)
        # Scale to training resolution
        dx = np.clip(bx + self._walk_x, -50, 50) * self._motion_scale
        dy = np.clip(by + self._walk_y, -20, 20) * self._motion_scale
        return dx, dy

    def _generate_rotation(self, t):
        return self._rot_amp * np.sin(2 * np.pi * self._rot_freq * t)

    # ── Render ────────────────────────────────────────────────────────────

    def _render_frame(self, sway_x, sway_y, rot_deg):
        frame = self._base_texture.copy()
        if abs(sway_x) > 0.01 or abs(sway_y) > 0.01:
            M = np.float32([[1, 0, sway_x], [0, 1, sway_y]])
            frame = cv2.warpAffine(frame, M, (self.frame_w, self.frame_h))
        if abs(rot_deg) > 0.001:
            cx, cy = self.frame_w // 2, self.frame_h // 2
            R = cv2.getRotationMatrix2D((cx, cy), rot_deg, 1.0)
            frame = cv2.warpAffine(frame, R, (self.frame_w, self.frame_h))
        noise = self._rng.normal(0, 2.0, frame.shape).astype(np.float32)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return frame

    # ── Phase correlation ─────────────────────────────────────────────────

    def _phase_correlate(self, frame):
        """Sub-pixel translation estimation via phase correlation."""
        ref_f = self._ref_gray.astype(np.float64)
        cur_f = frame.astype(np.float64)

        # Apply Hanning window to reduce edge effects
        hann_y = cv2.createHanningWindow((self.frame_w, self.frame_h), cv2.CV_64F)
        shift, response = cv2.phaseCorrelate(ref_f * hann_y, cur_f * hann_y)

        return shift[0], shift[1]  # (dx, dy) sub-pixel shift

    # ── Frame difference (captures rotation/distortion patterns) ──────────

    def _compute_diff(self, frame):
        """Downsampled absolute difference between reference and current.
        After phase-correlation translation is removed, what remains is
        rotation + distortion + noise — this is what the CNN learns from."""

        # Shift current frame by negated phase correlation to remove translation
        shifted = frame.copy()
        if abs(self._phase_dx) > 0.01 or abs(self._phase_dy) > 0.01:
            M = np.float32([[1, 0, -self._phase_dx], [0, 1, -self._phase_dy]])
            # This shifts the frame back to align with reference
            shifted = cv2.warpAffine(frame, M, (self.frame_w, self.frame_h))

        # Absolute difference
        diff = cv2.absdiff(self._ref_gray, shifted).astype(np.float32)

        # Downsample
        diff_small = cv2.resize(diff, (self.diff_w, self.diff_h),
                                interpolation=cv2.INTER_AREA)

        # Normalize to [0, 1]
        diff_max = diff_small.max()
        if diff_max > 0:
            diff_small /= diff_max

        return diff_small

    # ── Gym interface ─────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step = 0
        self._walk_x = 0.0
        self._walk_y = 0.0

        if self.domain_randomize:
            self._sway_amp_x = self._rng.uniform(10.0, 50.0)
            self._sway_amp_y = self._rng.uniform(1.0, 8.0)
            self._sway_freq_x = self._rng.uniform(0.05, 0.20)
            self._sway_freq_y = self._rng.uniform(0.05, 0.15)
            self._walk_sigma = self._rng.uniform(1.0, 5.0)
            self._walk_ar = self._rng.uniform(0.85, 0.98)
            self._rot_amp = self._rng.uniform(0.005, 0.05)
            self._rot_freq = self._rng.uniform(0.4, 1.2)
        else:
            self._sway_amp_x = 24.0
            self._sway_amp_y = 2.5
            self._sway_freq_x = 0.10
            self._sway_freq_y = 0.10
            self._walk_sigma = 3.0
            self._walk_ar = 0.95
            self._rot_amp = 0.012
            self._rot_freq = 0.8

        self._base_texture = self._make_texture()
        self._ref_gray = self._base_texture.copy()

        self._gt_sway_x, self._gt_sway_y = self._generate_sway(0)
        self._gt_rot = self._generate_rotation(0)
        current = self._render_frame(self._gt_sway_x, self._gt_sway_y, self._gt_rot)

        self._phase_dx, self._phase_dy = self._phase_correlate(current)
        self._frame_diff = self._compute_diff(current)

        self._prev_delta = np.zeros(4, dtype=np.float32)

        return self._build_obs(), {}

    def step(self, action):
        # Delta on top of phase correlation
        delta_tx    = float(action[0]) * self.delta_tx_max
        delta_ty    = float(action[1]) * self.delta_ty_max
        delta_theta = float(action[2]) * self.delta_theta_max
        delta_scale = float(action[3]) * self.delta_scale_max

        # Final correction = -phase_corr + delta
        # Phase correlation returns the shift (same direction as sway),
        # so negate it to get the correction
        act_tx    = -self._phase_dx + delta_tx
        act_ty    = -self._phase_dy + delta_ty
        act_theta = delta_theta       # phase corr gives no rotation
        act_scale = 1.0 + delta_scale

        # Residual
        residual_tx = self._gt_sway_x + act_tx
        residual_ty = self._gt_sway_y + act_ty
        residual_rot = self._gt_rot - act_theta

        # Reward
        accuracy = -(residual_tx ** 2 + residual_ty ** 2
                     + self.rotation_weight * residual_rot ** 2)
        delta_change = np.array([
            delta_tx - self._prev_delta[0] * self.delta_tx_max,
            delta_ty - self._prev_delta[1] * self.delta_ty_max,
        ], dtype=np.float32)
        smoothness = -self.smoothness_weight * float(np.sum(delta_change ** 2))
        scale_pen = -self.scale_weight * delta_scale ** 2

        reward = float(accuracy + smoothness + scale_pen)

        self._prev_delta = action.copy()

        # Advance
        self._step += 1
        t = self._step / self.fps
        self._gt_sway_x, self._gt_sway_y = self._generate_sway(t)
        self._gt_rot = self._generate_rotation(t)
        current = self._render_frame(self._gt_sway_x, self._gt_sway_y, self._gt_rot)
        self._phase_dx, self._phase_dy = self._phase_correlate(current)
        self._frame_diff = self._compute_diff(current)

        terminated = self._step >= self.episode_length
        return self._build_obs(), reward, terminated, False, {}

    def _build_obs(self):
        # Phase correlation (normalized)
        phase = np.array([self._phase_dx / self.tx_norm,
                          self._phase_dy / self.ty_norm], dtype=np.float32)

        # Patch statistics (centre 32×32 region of ref vs current)
        cx, cy = self.frame_w // 2, self.frame_h // 2
        r = 16
        ref_patch = self._ref_gray[cy-r:cy+r, cx-r:cx+r].astype(np.float32)
        # Approximate current patch stats from the frame diff
        diff_centre = self._frame_diff[self.diff_h//2-2:self.diff_h//2+2,
                                        self.diff_w//2-2:self.diff_w//2+2]
        patch_stats = np.array([
            ref_patch.mean() / 255.0, ref_patch.std() / 255.0,
            diff_centre.mean(), diff_centre.std()
        ], dtype=np.float32)

        time_norm = np.float32(self._step / self.episode_length)

        # Frame difference flattened (captures rotation/distortion)
        diff_flat = self._frame_diff.flatten().astype(np.float32)

        return np.concatenate([
            phase,                    # 2: phase correlation result
            patch_stats,              # 4: patch statistics
            self._prev_delta,         # 4: previous delta action
            [time_norm],              # 1: time
            diff_flat,                # diff_w × diff_h: frame difference
        ])


if __name__ == "__main__":
    import time

    env = FlowStabilizerEnv(domain_randomize=False, seed=42)
    print(f"Obs dim: {env.observation_space.shape[0]}")
    print(f"  Phase corr: 2")
    print(f"  Patch stats: 4")
    print(f"  Prev delta: 4")
    print(f"  Time: 1")
    print(f"  Frame diff: {env.diff_w}×{env.diff_h} = {env.diff_size}")
    print(f"  Total: {2 + 4 + 4 + 1 + env.diff_size}")
    print(f"Action: delta ±{env.delta_tx_max}px, ±{env.delta_theta_max}°")
    print()

    obs, _ = env.reset()
    print(f"Obs shape: {obs.shape}")

    # Speed
    start = time.time()
    for i in range(50):
        obs, r, done, _, _ = env.step(env.action_space.sample())
        if done: obs, _ = env.reset()
    fps_val = 50 / (time.time() - start)
    print(f"Speed: {fps_val:.0f} steps/sec")
    print()

    # Zero delta = phase correlation only
    obs, _ = env.reset(seed=42)
    total = 0
    for _ in range(300):
        obs, r, done, _, _ = env.step(np.zeros(4, dtype=np.float32))
        total += r
        if done: break
    print(f"Zero delta (phase corr only): reward={total:.0f}")
    print(f"  Per-frame residual: {np.sqrt(-total/300):.2f}px")

    # Oracle
    obs, _ = env.reset(seed=42)
    total = 0
    for _ in range(300):
        perfect_tx = -(env._gt_sway_x + env._phase_dx)
        perfect_ty = -(env._gt_sway_y + env._phase_dy)
        action = np.array([
            np.clip(perfect_tx / env.delta_tx_max, -1, 1),
            np.clip(perfect_ty / env.delta_ty_max, -1, 1),
            np.clip(env._gt_rot / env.delta_theta_max, -1, 1),
            0.0,
        ], dtype=np.float32)
        obs, r, done, _, _ = env.step(action)
        total += r
        if done: break
    print(f"Oracle delta: reward={total:.0f}")
    print(f"  Per-frame residual: {np.sqrt(-total/300):.3f}px")
