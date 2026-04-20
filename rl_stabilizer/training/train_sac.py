#!/usr/bin/env python3
"""
Train SAC — Phase correlation base + MLP delta refinement.

The observation is compact enough for a plain MLP (no CNN needed):
- Phase correlation shift (2)
- Patch statistics (4)
- Previous delta (4)
- Time (1)
- Frame difference (60×45 = 2700)
Total: ~2711 floats

Usage:
    python -m rl_stabilizer.training.train_sac
    python -m rl_stabilizer.training.train_sac --timesteps 500000 --name phase_delta
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from rl_stabilizer.envs.flow_stabilizer_env import FlowStabilizerEnv


def make_env(seed, domain_randomize=True):
    def _init():
        return FlowStabilizerEnv(domain_randomize=domain_randomize, seed=seed)
    return _init

def make_eval_env(seed):
    def _init():
        return FlowStabilizerEnv(domain_randomize=False, seed=seed)
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train SAC (phase correlation + delta)")
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--name", type=str, default="phase_delta")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n_envs", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parents[1] / "runs" / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating {args.n_envs} training envs + 1 eval env...")
    train_envs = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
    eval_env = SubprocVecEnv([make_eval_env(999)])

    # Plain MLP — no CNN needed. Phase correlation handles translation,
    # the MLP reads the frame difference to detect rotation patterns.
    policy_kwargs = dict(
        net_arch=[256, 256, 128],
        activation_fn=nn.ReLU,
    )

    print("Initializing SAC with MLP policy...")
    model = SAC(
        "MlpPolicy",
        train_envs,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        buffer_size=500000,
        batch_size=256,
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
        name_prefix="sac",
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
