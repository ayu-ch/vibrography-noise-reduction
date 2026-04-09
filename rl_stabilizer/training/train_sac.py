#!/usr/bin/env python3
"""
Train SAC policy for camera stabilization.

Usage:
    python -m rl_stabilizer.training.train_sac
    python -m rl_stabilizer.training.train_sac --timesteps 500000 --name quick_test
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from rl_stabilizer.envs.stabilizer_env import StabilizerEnv


def make_env(seed, domain_randomize=True):
    def _init():
        return StabilizerEnv(domain_randomize=domain_randomize, seed=seed)
    return _init


def make_eval_env(seed):
    """Eval env: no domain randomization, fixed parameters."""
    def _init():
        return StabilizerEnv(domain_randomize=False, seed=seed)
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train SAC stabilizer")
    parser.add_argument("--config", type=str,
                        default=str(Path(__file__).resolve().parents[1] /
                                    "configs" / "default.yaml"))
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override total_timesteps")
    parser.add_argument("--name", type=str, default="sac_stabilizer",
                        help="Experiment name")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    sac_cfg = cfg["sac"]
    pol_cfg = cfg["policy"]
    total_timesteps = args.timesteps or sac_cfg["total_timesteps"]
    n_envs = sac_cfg["n_envs"]

    output_dir = Path(__file__).resolve().parents[1] / "runs" / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Environments ──────────────────────────────────────────────────────
    print(f"Creating {n_envs} training envs + 1 eval env...")
    train_envs = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    eval_env = SubprocVecEnv([make_eval_env(999)])

    # ── Policy kwargs ─────────────────────────────────────────────────────
    policy_kwargs = dict(
        net_arch=pol_cfg["net_arch"],
        activation_fn=torch.nn.ReLU,
    )

    # ── SAC Model ─────────────────────────────────────────────────────────
    print(f"Initializing SAC: {pol_cfg['net_arch']}, lr={sac_cfg['learning_rate']}")
    model = SAC(
        "MlpPolicy",
        train_envs,
        policy_kwargs=policy_kwargs,
        learning_rate=sac_cfg["learning_rate"],
        buffer_size=sac_cfg["buffer_size"],
        batch_size=sac_cfg["batch_size"],
        gamma=sac_cfg["gamma"],
        tau=sac_cfg["tau"],
        ent_coef=sac_cfg["ent_coef"],
        learning_starts=sac_cfg["learning_starts"],
        train_freq=sac_cfg["train_freq"],
        verbose=1,
        tensorboard_log=str(output_dir / "tb_logs"),
        device=args.device,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq=max(50000 // n_envs, 1),
        save_path=str(output_dir / "checkpoints"),
        name_prefix="sac",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=max(25000 // n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"Training for {total_timesteps:,} steps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    # ── Save ──────────────────────────────────────────────────────────────
    final_path = output_dir / "final_model"
    model.save(str(final_path))
    print(f"Saved final model → {final_path}")

    train_envs.close()
    eval_env.close()


if __name__ == "__main__":
    main()
