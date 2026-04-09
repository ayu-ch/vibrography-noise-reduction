#!/usr/bin/env python3
"""
Benchmark RL policy vs Kalman filter on identical test sequences.

Usage:
    python -m rl_stabilizer.evaluation.benchmark --model runs/sac_stabilizer/best/best_model
    python -m rl_stabilizer.evaluation.benchmark --model runs/sac_stabilizer/best/best_model --episodes 20
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
from rl_stabilizer.evaluation.kalman_baseline import MotionKalman


def run_episode(env, policy_fn, seed):
    """Run one episode, return per-frame metrics."""
    obs, _ = env.reset(seed=seed)
    results = {"residual_tx": [], "residual_ty": [], "residual_rot": [],
               "action_tx": [], "action_ty": [], "gt_sway_x": [], "gt_sway_y": [],
               "gt_rot": []}

    done = False
    while not done:
        gt_sx = env._gt_sway_x
        gt_sy = env._gt_sway_y
        gt_r = env._gt_rot

        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Decode action to physical units
        act_tx = float(action[0]) * env.tx_max
        act_ty = float(action[1]) * env.ty_max
        act_theta = float(action[2]) * env.theta_max

        results["residual_tx"].append(gt_sx + act_tx)
        results["residual_ty"].append(gt_sy + act_ty)
        results["residual_rot"].append(gt_r - act_theta)
        results["action_tx"].append(act_tx)
        results["action_ty"].append(act_ty)
        results["gt_sway_x"].append(gt_sx)
        results["gt_sway_y"].append(gt_sy)
        results["gt_rot"].append(gt_r)

    return {k: np.array(v) for k, v in results.items()}


def kalman_policy_fn(kalman, env):
    """Wrap Kalman filter as a policy function returning [-1,1] actions."""
    def _fn(obs):
        # Extract current measurement from obs (first 4 values, denormalized)
        meas_tx = obs[0] * env.norm_t
        meas_ty = obs[1] * env.norm_t
        meas_theta = obs[2] * env.norm_r
        meas_scale = obs[3] + 1.0
        valid = obs[5] > 0.5
        inliers = int(obs[4] * env.inlier_saturate)

        if valid:
            state = kalman.update(meas_tx, meas_ty, meas_theta, meas_scale, inliers)
        else:
            state = kalman.predict_only()

        # Convert to [-1, 1] action space
        return np.array([
            np.clip(state[0] / env.tx_max, -1, 1),
            np.clip(state[1] / env.ty_max, -1, 1),
            np.clip(state[2] / env.theta_max, -1, 1),
            np.clip((state[3] - 1.0) / env.scale_half, -1, 1),
        ], dtype=np.float32)
    return _fn


def rms(arr):
    return float(np.sqrt(np.mean(arr ** 2)))


def main():
    parser = argparse.ArgumentParser(description="Benchmark RL vs Kalman")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained SAC model (without .zip)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of test episodes")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else \
        Path(args.model).parent.parent / "benchmark.png"

    # Load RL model
    print(f"Loading RL model: {args.model}")
    rl_model = SAC.load(args.model)

    def rl_policy_fn(obs):
        action, _ = rl_model.predict(obs, deterministic=True)
        return action

    # Run episodes
    env = StabilizerEnv(domain_randomize=False)

    rl_metrics = {"residual_tx": [], "residual_ty": [], "residual_rot": []}
    kf_metrics = {"residual_tx": [], "residual_ty": [], "residual_rot": []}

    # For plotting: save last episode data
    rl_last = kf_last = None

    for ep in range(args.episodes):
        seed = 10000 + ep
        print(f"  Episode {ep+1}/{args.episodes} (seed={seed})")

        # RL
        rl_res = run_episode(env, rl_policy_fn, seed)
        rl_metrics["residual_tx"].append(rms(rl_res["residual_tx"]))
        rl_metrics["residual_ty"].append(rms(rl_res["residual_ty"]))
        rl_metrics["residual_rot"].append(rms(rl_res["residual_rot"]))
        rl_last = rl_res

        # Kalman (fresh filter per episode)
        kf = MotionKalman()
        kf_fn = kalman_policy_fn(kf, env)
        kf_res = run_episode(env, kf_fn, seed)
        kf_metrics["residual_tx"].append(rms(kf_res["residual_tx"]))
        kf_metrics["residual_ty"].append(rms(kf_res["residual_ty"]))
        kf_metrics["residual_rot"].append(rms(kf_res["residual_rot"]))
        kf_last = kf_res

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'Metric':<25} {'Kalman':>12} {'RL (SAC)':>12} {'Improvement':>12}")
    print("-" * 60)
    for key, label in [("residual_tx", "Residual X RMS (px)"),
                       ("residual_ty", "Residual Y RMS (px)"),
                       ("residual_rot", "Residual θ RMS (°)")]:
        kf_val = np.mean(kf_metrics[key])
        rl_val = np.mean(rl_metrics[key])
        pct = (1 - rl_val / kf_val) * 100 if kf_val > 0 else 0
        print(f"{label:<25} {kf_val:>12.4f} {rl_val:>12.4f} {pct:>+11.1f}%")
    print("=" * 60)

    # ── Plot last episode comparison ──────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("RL (SAC) vs Kalman — Stabilization Residuals", fontsize=13)
    frames = np.arange(len(rl_last["residual_tx"]))

    for ax, key, label, unit in [
        (axes[0], "residual_tx", "Residual X", "px"),
        (axes[1], "residual_ty", "Residual Y", "px"),
        (axes[2], "residual_rot", "Residual θ", "°"),
    ]:
        kf_data = kf_last[key]
        rl_data = rl_last[key]
        ax.plot(frames, kf_data, color="steelblue", alpha=0.7, lw=0.8,
                label=f"Kalman (RMS {rms(kf_data):.3f} {unit})")
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
