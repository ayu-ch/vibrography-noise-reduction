#!/usr/bin/env python3
"""
Train SAC v2 — Full RANSAC replacement using optical flow input.

The policy uses a CNN to process the downsampled flow field, followed
by an MLP to output [tx, ty, θ, scale].

Usage:
    python -m rl_stabilizer.training.train_sac_v2
    python -m rl_stabilizer.training.train_sac_v2 --timesteps 500000 --name flow_test
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from rl_stabilizer.envs.flow_stabilizer_env import FlowStabilizerEnv


# ── Custom CNN feature extractor for optical flow ────────────────────────────

class FlowCNN(BaseFeaturesExtractor):
    """
    Extracts features from the flattened flow + action + time observation.

    1. Reshape the first (flow_w × flow_h × 2) elements into a 2-channel image
    2. Pass through 3 conv layers → flatten → 256-dim feature vector
    3. Concat with prev_action (4) and time (1) → 261-dim output
    """

    def __init__(self, observation_space: gym.spaces.Box,
                 flow_w=60, flow_h=45, features_dim=261):
        super().__init__(observation_space, features_dim)
        self.flow_w = flow_w
        self.flow_h = flow_h
        self.flow_size = flow_w * flow_h * 2

        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )

        # Compute CNN output dim
        with torch.no_grad():
            dummy = torch.zeros(1, 2, flow_h, flow_w)
            cnn_out = self.cnn(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(cnn_out, 256),
            nn.ReLU(),
        )

        # Final features = 256 (from CNN) + 4 (prev_action) + 1 (time) = 261
        self._features_dim = 256 + 5

    @property
    def features_dim(self):
        return self._features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]

        # Split observation into flow + extras
        flow_flat = observations[:, :self.flow_size]
        extras = observations[:, self.flow_size:]  # prev_action(4) + time(1)

        # Reshape flow to (batch, 2, H, W)
        flow_2d = flow_flat.reshape(batch_size, self.flow_h, self.flow_w, 2)
        flow_2d = flow_2d.permute(0, 3, 1, 2)  # → (batch, 2, H, W)

        # CNN
        cnn_features = self.fc(self.cnn(flow_2d))  # → (batch, 256)

        # Concat with extras
        return torch.cat([cnn_features, extras], dim=1)  # → (batch, 261)


# ── Environment factories ─────────────────────────────────────────────────────

def make_env(seed, domain_randomize=True):
    def _init():
        return FlowStabilizerEnv(domain_randomize=domain_randomize, seed=seed)
    return _init

def make_eval_env(seed):
    def _init():
        return FlowStabilizerEnv(domain_randomize=False, seed=seed)
    return _init


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train SAC v2 (flow-based)")
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--name", type=str, default="flow_sac")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n_envs", type=int, default=4,
                        help="Parallel envs (lower than v1 because flow is slower)")
    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parents[1] / "runs" / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating {args.n_envs} training envs + 1 eval env...")
    train_envs = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
    eval_env = SubprocVecEnv([make_eval_env(999)])

    policy_kwargs = dict(
        features_extractor_class=FlowCNN,
        features_extractor_kwargs=dict(flow_w=60, flow_h=45),
        net_arch=[256, 128],  # MLP after CNN features
        activation_fn=nn.ReLU,
    )

    print(f"Initializing SAC with FlowCNN extractor...")
    model = SAC(
        "MlpPolicy",
        train_envs,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        buffer_size=500000,      # smaller buffer (each obs is 5405 floats)
        batch_size=128,          # smaller batch (larger obs)
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        learning_starts=5000,
        train_freq=1,
        verbose=1,
        tensorboard_log=str(output_dir / "tb_logs"),
        device=args.device,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(25000 // args.n_envs, 1),
        save_path=str(output_dir / "checkpoints"),
        name_prefix="sac_flow",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=max(10000 // args.n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
    )

    print(f"Training for {args.timesteps:,} steps...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    final_path = output_dir / "final_model"
    model.save(str(final_path))
    print(f"Saved → {final_path}")

    train_envs.close()
    eval_env.close()


if __name__ == "__main__":
    main()
