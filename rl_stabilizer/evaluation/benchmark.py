#!/usr/bin/env python3
"""
Benchmark RL policy vs raw RANSAC on identical test sequences.

Usage:
    python -m rl_stabilizer.evaluation.benchmark --model runs/real_calibrated/best/best_model
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stable_baselines3 import SAC
from rl_stabilizer.envs.stabilizer_env import StabilizerEnv


def run_episode_rl(env, rl_model, seed):
    """Run one episode with the RL policy."""
    obs, _ = env.reset(seed=seed)
    results = {"residual_tx": [], "residual_ty": [], "residual_rot": []}

    done = False
    while not done:
        gt_sx = env._gt_sway_x
        gt_sy = env._gt_sway_y
        gt_r = env._gt_rot

        action, _ = rl_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # The env computes the final correction internally:
        # act_tx = meas_tx + delta_tx (RANSAC + agent's refinement)
        # Residual = gt_sway + act_tx
        # We can read it from the reward components, but simpler to recompute:
        delta_tx = float(action[0]) * env.delta_tx_max
        delta_ty = float(action[1]) * env.delta_ty_max
        delta_theta = float(action[2]) * env.delta_theta_max

        # Get the RANSAC measurement that was active during this step
        # (it was set before step() advanced to next frame)
        # We need to reconstruct: act = meas + delta, residual = gt + act
        # Since env already advanced, we use the stored prev values
        act_tx = env._prev_action[0]
        act_ty = env._prev_action[1]
        act_theta = env._prev_action[2]

        results["residual_tx"].append(gt_sx + act_tx)
        results["residual_ty"].append(gt_sy + act_ty)
        results["residual_rot"].append(gt_r - act_theta)

    return {k: np.array(v) for k, v in results.items()}


def run_episode_raw_ransac(env, seed):
    """Run one episode using raw RANSAC measurement directly (no RL, no filter).
    This is what ransac_stabilizer.cpp does — use H directly."""
    obs, _ = env.reset(seed=seed)
    results = {"residual_tx": [], "residual_ty": [], "residual_rot": []}

    done = False
    while not done:
        gt_sx = env._gt_sway_x
        gt_sy = env._gt_sway_y
        gt_r = env._gt_rot

        # Raw RANSAC = use measurement directly, delta = 0
        action = np.zeros(4, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # With delta=0, act = meas + 0 = meas
        act_tx = env._prev_action[0]
        act_ty = env._prev_action[1]
        act_theta = env._prev_action[2]

        results["residual_tx"].append(gt_sx + act_tx)
        results["residual_ty"].append(gt_sy + act_ty)
        results["residual_rot"].append(gt_r - act_theta)

    return {k: np.array(v) for k, v in results.items()}


def rms(arr):
    return float(np.sqrt(np.mean(arr ** 2)))


def main():
    parser = argparse.ArgumentParser(description="Benchmark RL vs Raw RANSAC")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained SAC model")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else \
        Path(args.model).parent.parent / "benchmark.png"

    print(f"Loading RL model: {args.model}")
    rl_model = SAC.load(args.model)

    env = StabilizerEnv(domain_randomize=False)

    rl_metrics = {"residual_tx": [], "residual_ty": [], "residual_rot": []}
    raw_metrics = {"residual_tx": [], "residual_ty": [], "residual_rot": []}

    rl_last = raw_last = None

    for ep in range(args.episodes):
        seed = 10000 + ep
        print(f"  Episode {ep+1}/{args.episodes} (seed={seed})")

        # RL
        rl_res = run_episode_rl(env, rl_model, seed)
        rl_metrics["residual_tx"].append(rms(rl_res["residual_tx"]))
        rl_metrics["residual_ty"].append(rms(rl_res["residual_ty"]))
        rl_metrics["residual_rot"].append(rms(rl_res["residual_rot"]))
        rl_last = rl_res

        # Raw RANSAC (delta = 0, use measurement directly)
        raw_res = run_episode_raw_ransac(env, seed)
        raw_metrics["residual_tx"].append(rms(raw_res["residual_tx"]))
        raw_metrics["residual_ty"].append(rms(raw_res["residual_ty"]))
        raw_metrics["residual_rot"].append(rms(raw_res["residual_rot"]))
        raw_last = raw_res

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"{'Metric':<25} {'Raw RANSAC':>12} {'RL (SAC)':>12} {'Improvement':>12}")
    print("-" * 65)
    for key, label in [("residual_tx", "Residual X RMS (px)"),
                       ("residual_ty", "Residual Y RMS (px)"),
                       ("residual_rot", "Residual θ RMS (°)")]:
        raw_val = np.mean(raw_metrics[key])
        rl_val = np.mean(rl_metrics[key])
        pct = (1 - rl_val / raw_val) * 100 if raw_val > 0 else 0
        print(f"{label:<25} {raw_val:>12.4f} {rl_val:>12.4f} {pct:>+11.1f}%")
    print("=" * 65)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("RL (SAC) vs Raw RANSAC — Stabilization Residuals", fontsize=13)
    frames = np.arange(len(rl_last["residual_tx"]))

    for ax, key, label, unit in [
        (axes[0], "residual_tx", "Residual X", "px"),
        (axes[1], "residual_ty", "Residual Y", "px"),
        (axes[2], "residual_rot", "Residual θ", "°"),
    ]:
        raw_data = raw_last[key]
        rl_data = rl_last[key]
        ax.plot(frames, raw_data, color="steelblue", alpha=0.7, lw=0.8,
                label=f"Raw RANSAC (RMS {rms(raw_data):.3f} {unit})")
        ax.plot(frames, rl_data, color="firebrick", alpha=0.8, lw=0.9,
                label=f"RL SAC (RMS {rms(rl_data):.3f} {unit})")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_ylabel(f"{label} ({unit})", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Frame", fontsize=9)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {output_path}")


if __name__ == "__main__":
    main()
