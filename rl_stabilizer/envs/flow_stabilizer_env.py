"""
RL Environment — Mean Flow + CNN Delta

The agent does NOT output the full correction. Instead:
1. Compute dense optical flow between reference and current frame
2. Mean background flow = base correction (replaces RANSAC, ~90% accurate)
3. Subtract mean from flow → residual flow (spatial patterns: rotation, noise, vibration)
4. CNN reads the residual flow → outputs small delta [±3px, ±3px, ±0.05°, ±0.002]
5. Final correction = mean_flow + delta

This is the best of both worlds:
- Mean flow handles the large displacement (no learning needed)
- CNN only learns to fix what mean flow misses (small, structured residual)
- Zero delta = mean flow result (decent baseline from step 0 of training)

Observation:
    Residual flow (flow - mean) downsampled (flow_h × flow_w × 2), flattened
    + mean_flow_x, mean_flow_y (2)
    + prev_delta (4)
    + time (1)
    Total: flow_h × flow_w × 2 + 7

Action (4 floats in [-1, 1]):
    Delta on top of mean flow: [δtx, δty, δθ, δscale]
    Rescaled to ±3px, ±3px, ±0.05°, ±0.002
"""

import gymnasium as gym
import numpy as np
import cv2
from gymnasium import spaces


class FlowStabilizerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, domain_randomize=True, seed=None,
                 frame_w=480, frame_h=360,
                 flow_w=60, flow_h=45,
                 episode_length=300, fps=100):
        super().__init__()
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.flow_w = flow_w
        self.flow_h = flow_h
        self.episode_length = episode_length
        self.fps = fps
        self.domain_randomize = domain_randomize

        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

        # Observation: residual flow + mean_flow(2) + prev_delta(4) + time(1)
        self.flow_size = flow_w * flow_h * 2
        obs_dim = self.flow_size + 2 + 4 + 1
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        # Delta scaling (small adjustments on top of mean flow)
        self.delta_tx_max = 3.0
        self.delta_ty_max = 3.0
        self.delta_theta_max = 0.05   # degrees
        self.delta_scale_max = 0.002

        # For normalizing the residual flow in observations
        self.flow_norm = 5.0   # residual flow is small, ±5px max

        # Reward weights
        self.rotation_weight = 100.0
        self.smoothness_weight = 0.05
        self.scale_weight = 1000.0

        self._rng = np.random.default_rng(seed)
        self._base_texture = None

        # Try CUDA Farneback, fall back to CPU
        self._use_cuda = False
        try:
            self._farneback_gpu = cv2.cuda.FarnebackOpticalFlow.create()
            self._farneback_gpu.setNumLevels(3)
            self._farneback_gpu.setPyrScale(0.5)
            self._farneback_gpu.setWinSize(15)
            self._farneback_gpu.setNumIters(3)
            self._farneback_gpu.setPolyN(5)
            self._farneback_gpu.setPolySigma(1.2)
            self._use_cuda = True
            print("FlowStabilizerEnv: Using CUDA Farneback")
        except Exception:
            print("FlowStabilizerEnv: Using CPU Farneback")

    # ── Texture generation ────────────────────────────────────────────────

    def _make_texture(self):
        tex = np.full((self.frame_h, self.frame_w), 200, dtype=np.uint8)
        n = int(self.frame_w * self.frame_h * 0.06)
        xs = self._rng.integers(0, self.frame_w, size=n)
        ys = self._rng.integers(0, self.frame_h, size=n)
        vals = self._rng.integers(0, 180, size=n).astype(np.uint8)
        tex[ys, xs] = vals
        tex = cv2.GaussianBlur(tex, (3, 3), 0.8)
        return tex

    # ── Motion generators ─────────────────────────────────────────────────

    def _generate_sway(self, t):
        bx = self._sway_amp_x * np.sin(2 * np.pi * self._sway_freq_x * t)
        by = self._sway_amp_y * np.cos(2 * np.pi * self._sway_freq_y * t)
        self._walk_x = self._walk_x * self._walk_ar + self._rng.normal(0, self._walk_sigma)
        self._walk_y = self._walk_y * self._walk_ar + self._rng.normal(0, self._walk_sigma * 0.3)
        dx = np.clip(bx + self._walk_x, -50.0, 50.0)
        dy = np.clip(by + self._walk_y, -10.0, 10.0)
        return dx, dy

    def _generate_rotation(self, t):
        return self._rot_amp * np.sin(2 * np.pi * self._rot_freq * t)

    # ── Render frame ──────────────────────────────────────────────────────

    def _render_frame(self, sway_x, sway_y, rot_deg):
        if self._use_cuda:
            gpu_frame = cv2.cuda.GpuMat(self._base_texture)
            if abs(sway_x) > 0.01 or abs(sway_y) > 0.01:
                M = np.float32([[1, 0, sway_x], [0, 1, sway_y]])
                gpu_frame = cv2.cuda.warpAffine(gpu_frame, M,
                                                (self.frame_w, self.frame_h))
            if abs(rot_deg) > 0.001:
                cx, cy = self.frame_w // 2, self.frame_h // 2
                R = cv2.getRotationMatrix2D((cx, cy), rot_deg, 1.0)
                gpu_frame = cv2.cuda.warpAffine(gpu_frame, R,
                                                (self.frame_w, self.frame_h))
            frame = gpu_frame.download()
        else:
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

    # ── Compute flow and extract mean + residual ──────────────────────────

    def _compute_flow(self, frame):
        if self._use_cuda:
            gpu_ref = cv2.cuda.GpuMat(self._ref_gray)
            gpu_cur = cv2.cuda.GpuMat(frame)
            gpu_flow = self._farneback_gpu.calc(gpu_ref, gpu_cur, None)
            flow = gpu_flow.download()
        else:
            flow = cv2.calcOpticalFlowFarneback(
                self._ref_gray, frame, None,
                0.5, 3, 15, 3, 5, 1.2, 0)

        # Downsample
        flow_small = cv2.resize(flow, (self.flow_w, self.flow_h),
                                interpolation=cv2.INTER_AREA)

        # Mean flow = base correction (what RANSAC would give)
        mean_dx = float(flow_small[:, :, 0].mean())
        mean_dy = float(flow_small[:, :, 1].mean())

        # Residual flow = what mean flow misses (rotation, noise patterns)
        residual = flow_small.copy()
        residual[:, :, 0] -= mean_dx
        residual[:, :, 1] -= mean_dy

        # Normalize residual (it's small, typically ±2px)
        residual[:, :, 0] /= self.flow_norm
        residual[:, :, 1] /= self.flow_norm

        return residual, mean_dx, mean_dy

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

        self._residual_flow, self._mean_dx, self._mean_dy = \
            self._compute_flow(current)

        self._prev_delta = np.zeros(4, dtype=np.float32)

        return self._build_obs(), {}

    def step(self, action):
        # Delta on top of mean flow
        delta_tx    = float(action[0]) * self.delta_tx_max
        delta_ty    = float(action[1]) * self.delta_ty_max
        delta_theta = float(action[2]) * self.delta_theta_max
        delta_scale = float(action[3]) * self.delta_scale_max

        # Final correction = mean flow + delta
        # Mean flow ≈ -sway (flow points opposite to motion)
        act_tx    = self._mean_dx + delta_tx
        act_ty    = self._mean_dy + delta_ty
        act_theta = delta_theta      # mean flow has no rotation info
        act_scale = 1.0 + delta_scale

        # Residual = gt_sway + correction (should be ~0)
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

        # Store normalized delta for history
        self._prev_delta = action.copy()

        # Advance
        self._step += 1
        t = self._step / self.fps
        self._gt_sway_x, self._gt_sway_y = self._generate_sway(t)
        self._gt_rot = self._generate_rotation(t)
        current = self._render_frame(self._gt_sway_x, self._gt_sway_y, self._gt_rot)
        self._residual_flow, self._mean_dx, self._mean_dy = \
            self._compute_flow(current)

        terminated = self._step >= self.episode_length
        return self._build_obs(), reward, terminated, False, {}

    def _build_obs(self):
        flow_flat = self._residual_flow.flatten().astype(np.float32)
        time_norm = np.float32(self._step / self.episode_length)
        return np.concatenate([
            flow_flat,                                          # residual flow
            [self._mean_dx / 50.0, self._mean_dy / 10.0],     # normalized mean flow
            self._prev_delta,                                   # prev delta action
            [time_norm],
        ])


if __name__ == "__main__":
    import time

    env = FlowStabilizerEnv(domain_randomize=False, seed=42)
    print(f"Obs dim: {env.observation_space.shape[0]}")
    print(f"  Residual flow: {env.flow_w}x{env.flow_h}x2 = {env.flow_size}")
    print(f"  + mean_flow(2) + prev_delta(4) + time(1) = {env.flow_size + 7}")
    print(f"Action dim: {env.action_space.shape[0]}  (delta ±{env.delta_tx_max}px)")
    print()

    obs, _ = env.reset()
    print(f"Obs shape: {obs.shape}")

    # Speed test
    start = time.time()
    for i in range(50):
        obs, r, done, _, _ = env.step(env.action_space.sample())
        if done: obs, _ = env.reset()
    fps = 50 / (time.time() - start)
    print(f"Speed: {fps:.1f} steps/sec")
    print()

    # Zero delta = mean flow only
    obs, _ = env.reset(seed=42)
    total = 0
    for _ in range(100):
        obs, r, done, _, _ = env.step(np.zeros(4, dtype=np.float32))
        total += r
    print(f"Zero delta (mean flow only): reward={total:.0f}")
    print(f"  Per-frame residual ≈ {np.sqrt(-total/100):.2f}px")

    # Oracle
    obs, _ = env.reset(seed=42)
    total = 0
    for _ in range(100):
        # Perfect delta = exactly what mean flow missed
        perfect_tx = -(env._gt_sway_x + env._mean_dx)
        perfect_ty = -(env._gt_sway_y + env._mean_dy)
        action = np.array([
            np.clip(perfect_tx / env.delta_tx_max, -1, 1),
            np.clip(perfect_ty / env.delta_ty_max, -1, 1),
            np.clip(env._gt_rot / env.delta_theta_max, -1, 1),
            0.0,
        ], dtype=np.float32)
        obs, r, done, _, _ = env.step(action)
        total += r
    print(f"Oracle delta: reward={total:.0f}")
    print(f"  Per-frame residual ≈ {np.sqrt(-total/100):.3f}px")
