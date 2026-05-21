"""
Re-render overlay.mp4 from an existing poses.csv + frame directory.

Used when the in-line overlay from blade_tracker_sam2.py failed (e.g. mp4v codec
choking on long 4K streams) but the CSV is still good.

Usage:
    python render_overlay.py \
        --frame-dir c0282_frames \
        --poses c0282_sam2_single_full/poses.csv \
        --out  c0282_sam2_single_full/overlay_h264.mp4 \
        --fps 119.88
"""

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def draw_overlay_from_row(frame, row, n_blades: int):
    """Mimics blade_tracker.draw_overlay but reads a CSV row."""
    out = frame.copy()
    if not row["pose_valid"]:
        cv2.putText(out, f"frame {int(row['frame_id'])} (no pose)",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return out
    hub = (int(round(row["hub_x"])), int(round(row["hub_y"])))
    cv2.circle(out, hub, 10, (0, 0, 255), 2)
    cv2.drawMarker(out, hub, (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
    palette = [(0, 255, 0), (255, 255, 0), (255, 0, 255), (255, 128, 0)]
    for k in range(n_blades):
        tx = row.get(f"b{k}_tip_x")
        ty = row.get(f"b{k}_tip_y")
        ang = row.get(f"b{k}_angle_deg")
        if pd.isna(tx) or pd.isna(ty):
            continue
        color = palette[k % len(palette)]
        tip = (int(round(tx)), int(round(ty)))
        cv2.line(out, hub, tip, color, 2)
        cv2.circle(out, tip, 6, color, 2)
        cv2.putText(out, f"B{k} {ang:+.1f}",
                    (tip[0] + 8, tip[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.putText(out, f"frame {int(row['frame_id'])}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-dir", required=True)
    ap.add_argument("--poses", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="scale factor for output video (0.5 = half-res, smaller file)")
    ap.add_argument("--codec", default="avc1",
                    help="FourCC code: avc1 (H.264, default), mp4v, X264, mjpg")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.poses)
    n_blades = sum(1 for c in df.columns if c.startswith("b") and c.endswith("_tip_x"))
    print(f"poses: {len(df)} rows  blades: {n_blades}")

    frame_dir = Path(args.frame_dir)
    files = sorted([p for p in frame_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if args.max_frames > 0:
        files = files[:args.max_frames]
    n_files = len(files)
    n_pairs = min(n_files, len(df))
    print(f"frames: {n_files}  rendering: {n_pairs}")

    writer = None
    t0 = time.monotonic()
    for i in range(n_pairs):
        frame = cv2.imread(str(files[i]))
        if frame is None:
            print(f"  skipping unreadable {files[i]}", file=sys.stderr)
            continue
        row = df.iloc[i]
        vis = draw_overlay_from_row(frame, row, n_blades)
        if args.scale != 1.0:
            h, w = vis.shape[:2]
            vis = cv2.resize(vis, (int(w * args.scale), int(h * args.scale)))
        if writer is None:
            h, w = vis.shape[:2]
            writer = cv2.VideoWriter(
                args.out,
                cv2.VideoWriter_fourcc(*args.codec),
                args.fps, (w, h),
            )
            if not writer.isOpened():
                raise SystemExit(f"VideoWriter failed to open with codec {args.codec}. "
                                 "Try --codec mp4v or --codec mjpg.")
            print(f"writer: {w}x{h} @ {args.fps} fps  codec={args.codec}")
        writer.write(vis)
        if (i + 1) % 60 == 0 or i + 1 == n_pairs:
            elapsed = time.monotonic() - t0
            fps = (i + 1) / max(1e-3, elapsed)
            eta = (n_pairs - i - 1) / max(1e-3, fps)
            sys.stdout.write(
                f"\r[ {i+1:>6d} / {n_pairs} ]  {100*(i+1)/n_pairs:5.1f}%  "
                f"{fps:6.1f} fps  ETA {time.strftime('%H:%M:%S', time.gmtime(eta))}"
            )
            sys.stdout.flush()

    if writer is not None:
        writer.release()
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
