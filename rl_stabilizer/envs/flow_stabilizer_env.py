"""
RL Environment v2 — Full RANSAC Replacement

The agent sees DENSE OPTICAL FLOW between the current frame and the
reference frame, and directly outputs the stabilization correction
[tx, ty, θ, scale]. No ORB, no RANSAC, no feature matching.

Pipeline:
    frame_t + frame_ref → Farneback optical flow → downsample
    → CNN policy → [tx, ty, θ, scale] → warpPerspective

Training flow:
    1. Generate synthetic frame with known camera sway
    2. Compute optical flow between swayed frame and reference
    3. Agent sees the flow, outputs correction
    4. Reward = how well the correction cancels the sway

The optical flow is RENDERED during training (unlike v1 which simulated
RANSAC output with math). This is slower but the agent learns to read
actual motion patterns, not just numbers.

Observation:
    Downsampled optical flow (flow_h × flow_w × 2) flattened
    + [prev_action (4), frame_count_normalized (1)] = total obs dim

Action (4 floats, continuous):
    [tx, ty, θ, scale] — the full correction to apply
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
        """
        Args:
            frame_w, frame_h: resolution for synthetic frame rendering
                              (smaller than real 1920×1080 for fast training)
            flow_w, flow_h:   downsampled flow resolution for observation
            episode_length:   frames per episode
        """
        super().__init__()
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.flow_w = flow_w
        self.flow_h = flow_h
        self.episode_length = episode_length
        self.fps = fps
        self.domain_randomize = domain_randomize

        # Action: [tx, ty, θ, scale] in [-1, 1]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

        # Observation: flattened downsampled flow (2 channels) + prev_action + frame_count
        self.flow_size = flow_w * flow_h * 2
        obs_dim = self.flow_size + 4 + 1  # flow + prev_action + time
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        # Action scaling (real video stats)
        self.tx_max = 50.0
        self.ty_max = 10.0
        self.theta_max = 0.1      # degrees
        self.scale_half = 0.005

        # Reward weights
        self.rotation_weight = 100.0
        self.smoothness_weight = 0.05
        self.scale_weight = 1000.0

        self._rng = np.random.default_rng(seed)
        self._base_texture = None

        # Try CUDA Farneback (10x faster on DGX), fall back to CPU
        self._use_cuda = False
        try:
            self._farneback_gpu = cv2.cuda.FarnebackOpticalFlow.create(
                numLevels=3, pyrScale=0.5, winSize=15,
                numIters=3, polyN=5, polySigma=1.2, flags=0)
            self._use_cuda = True
            print("FlowStabilizerEnv: Using CUDA Farneback")
        except Exception:
            print("FlowStabilizerEnv: CUDA not available, using CPU Farneback")

    # ── Texture generation (simplified speckle) ───────────────────────────

    def _make_texture(self):
        """Generate a speckle texture at training resolution."""
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
        dx = np.clip(bx + self._walk_x, -self.tx_max, self.tx_max)
        dy = np.clip(by + self._walk_y, -self.ty_max, self.ty_max)
        return dx, dy

    def _generate_rotation(self, t):
        return self._rot_amp * np.sin(2 * np.pi * self._rot_freq * t)

    # ── Apply motion to texture → get "current frame" ─────────────────────

    def _render_frame(self, sway_x, sway_y, rot_deg):
        """Apply sway + rotation to base texture → return swayed frame."""
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

        # Sensor noise (stays on CPU — small operation)
        noise = self._rng.normal(0, 2.0, frame.shape).astype(np.float32)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return frame

    # ── Compute optical flow ──────────────────────────────────────────────

    def _compute_flow(self, frame):
        """Farneback dense optical flow: current frame vs reference."""
        if self._use_cuda:
            gpu_ref = cv2.cuda.GpuMat(self._ref_gray)
            gpu_cur = cv2.cuda.GpuMat(frame)
            gpu_flow = self._farneback_gpu.calc(gpu_ref, gpu_cur, None)
            flow = gpu_flow.download()
        else:
            flow = cv2.calcOpticalFlowFarneback(
                self._ref_gray, frame,
                None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        # Downsample to observation size
        flow_small = cv2.resize(flow, (self.flow_w, self.flow_h),
                                interpolation=cv2.INTER_AREA)

        # Normalize: divide by max expected displacement
        flow_small[:, :, 0] /= self.tx_max   # x flow
        flow_small[:, :, 1] /= self.ty_max   # y flow

        return flow_small

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

        # Generate base texture (reference frame)
        self._base_texture = self._make_texture()
        self._ref_gray = self._base_texture.copy()

        # Generate first swayed frame
        self._gt_sway_x, self._gt_sway_y = self._generate_sway(0)
        self._gt_rot = self._generate_rotation(0)
        current = self._render_frame(self._gt_sway_x, self._gt_sway_y, self._gt_rot)

        # Compute flow
        self._flow = self._compute_flow(current)

        self._prev_action = np.zeros(4, dtype=np.float32)

        return self._build_obs(), {}

    def step(self, action):
        # Rescale action to physical units
        act_tx    = float(action[0]) * self.tx_max
        act_ty    = float(action[1]) * self.ty_max
        act_theta = float(action[2]) * self.theta_max
        act_scale = 1.0 + float(action[3]) * self.scale_half

        # Residual = how much sway remains after correction
        # Correction maps current→reference, so correction ≈ -sway ideally
        # H maps current→ref means tx_correction ≈ -sway_x
        residual_tx = self._gt_sway_x + act_tx
        residual_ty = self._gt_sway_y + act_ty
        residual_rot = self._gt_rot - act_theta

        # Reward
        accuracy = -(residual_tx ** 2 + residual_ty ** 2
                     + self.rotation_weight * residual_rot ** 2)

        delta = np.array([act_tx - self._prev_action[0] * self.tx_max,
                          act_ty - self._prev_action[1] * self.ty_max],
                         dtype=np.float32)
        smoothness = -self.smoothness_weight * float(np.sum(delta ** 2))

        scale_pen = -self.scale_weight * (act_scale - 1.0) ** 2

        reward = float(accuracy + smoothness + scale_pen)

        # Store action (normalized) for history
        self._prev_action = action.copy()

        # Advance to next frame
        self._step += 1
        t = self._step / self.fps
        self._gt_sway_x, self._gt_sway_y = self._generate_sway(t)
        self._gt_rot = self._generate_rotation(t)
        current = self._render_frame(self._gt_sway_x, self._gt_sway_y, self._gt_rot)
        self._flow = self._compute_flow(current)

        terminated = self._step >= self.episode_length
        return self._build_obs(), reward, terminated, False, {}

    def _build_obs(self):
        """Flatten flow + prev_action + time → observation vector."""
        flow_flat = self._flow.flatten().astype(np.float32)
        time_norm = np.float32(self._step / self.episode_length)
        obs = np.concatenate([
            flow_flat,
            self._prev_action,
            [time_norm],
        ])
        return obs


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import time

    env = FlowStabilizerEnv(domain_randomize=False, seed=42)

    print(f"Observation dim: {env.observation_space.shape[0]}")
    print(f"  Flow: {env.flow_w}x{env.flow_h}x2 = {env.flow_size}")
    print(f"  Prev action: 4")
    print(f"  Time: 1")
    print(f"  Total: {env.flow_size + 5}")
    print(f"Action dim: {env.action_space.shape[0]}")
    print()

    obs, _ = env.reset()
    print(f"Obs shape: {obs.shape}")

    # Benchmark speed
    start = time.time()
    n_steps = 50
    for i in range(n_steps):
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        if done:
            obs, _ = env.reset()
    elapsed = time.time() - start
    fps = n_steps / elapsed
    print(f"{n_steps} steps in {elapsed:.1f}s = {fps:.1f} steps/sec")
    print(f"(Training at 8 envs × {fps:.0f} fps = {8*fps:.0f} steps/sec)")
    print()

    # Test with zero action (no correction)
    obs, _ = env.reset(seed=42)
    total_reward = 0
    for i in range(100):
        obs, r, done, _, _ = env.step(np.zeros(4, dtype=np.float32))
        total_reward += r
    print(f"Zero action (no correction) reward/100 frames: {total_reward:.0f}")

    # Test with oracle action
    obs, _ = env.reset(seed=42)
    total_reward = 0
    for i in range(100):
        # Perfect correction: action = -sway / max
        action = np.array([
            -env._gt_sway_x / env.tx_max,
            -env._gt_sway_y / env.ty_max,
            env._gt_rot / env.theta_max,
            0.0,
        ], dtype=np.float32)
        action = np.clip(action, -1, 1)
        obs, r, done, _, _ = env.step(action)
        total_reward += r
    print(f"Oracle action (perfect correction) reward/100 frames: {total_reward:.0f}")
