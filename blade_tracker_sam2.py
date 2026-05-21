"""
SAM2-based blade tracker (Approach A: ML detection + blade-fixed coords).

Pipeline:
  1. Read prompts.json (one click per blade on a reference frame)
  2. Run SAM2 video propagation -> per-frame, per-blade binary mask
  3. PCA on each blade mask -> seed axis
  4. Hub = least-squares intersection of the three blade axes
  5. Tip per blade = farthest mask pixel from hub
  6. Write poses.csv + overlay.mp4

SAM2 itself is heavy — install separately on the DGX:
    pip install torch torchvision
    pip install git+https://github.com/facebookresearch/sam2.git
    # then download a checkpoint (e.g. sam2.1_hiera_large.pt) into ./sam2_checkpoints/

This script imports SAM2 lazily so it can be inspected on machines without it.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Reuse helpers from the classical tracker.
from blade_tracker import (
    BladeObs,
    FramePose,
    Progress,
    draw_overlay,
    intersect_axes,
    pca_axis,
)


def load_prompts(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if "prompts" not in payload or not payload["prompts"]:
        raise SystemExit(f"{path} has no prompts")
    return payload


def list_frames(frame_dir: Path) -> list[Path]:
    files = sorted([p for p in frame_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not files:
        raise SystemExit(f"no frames in {frame_dir}")
    return files


def build_sam2(checkpoint: Path, model_cfg: str, device: str):
    """Lazy import — only succeed if SAM2 is installed."""
    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as e:
        raise SystemExit(
            "SAM2 not installed. On the DGX run:\n"
            "    pip install torch torchvision\n"
            "    pip install git+https://github.com/facebookresearch/sam2.git\n"
            f"original error: {e}"
        )
    predictor = build_sam2_video_predictor(model_cfg, str(checkpoint), device=device)
    return predictor


def extract_pose_from_masks(
    masks_per_obj: dict[int, np.ndarray],
    min_area: int,
    fixed_hub: tuple[float, float] | None = None,
) -> FramePose | None:
    """
    masks_per_obj: { obj_id: HxW bool/uint8 mask }, one entry per prompted blade.
    fixed_hub:    if given, use this static (x,y) for every frame instead of
                  recomputing from axis intersection. REQUIRED in single-blade mode
                  (one axis has no unique intersection point).
    Returns FramePose with blades sorted by obj_id so column k always == blade k.
    """
    seeds: list[tuple[int, BladeObs, np.ndarray]] = []
    for obj_id, mask in masks_per_obj.items():
        ys, xs = np.where(mask > 0)
        if xs.size < min_area:
            continue
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        mean, axis, angle, half_len = pca_axis(pts)
        seed = BladeObs(
            centroid=(float(mean[0]), float(mean[1])),
            angle_deg=angle,
            tip=(float(mean[0] + axis[0] * half_len),
                 float(mean[1] + axis[1] * half_len)),
            length=half_len * 2.0,
            area=int(xs.size),
        )
        seeds.append((obj_id, seed, pts))

    if not seeds:
        return None

    seeds.sort(key=lambda t: t[0])
    seed_blades = [s for _, s, _ in seeds]
    if fixed_hub is not None:
        hub = fixed_hub
    elif len(seed_blades) >= 2:
        hub = intersect_axes(seed_blades)
    else:
        hub = seed_blades[0].centroid  # single-blade fallback if no fixed_hub

    refined: list[BladeObs] = []
    for _obj_id, seed, pts in seeds:
        d2 = (pts[:, 0] - hub[0]) ** 2 + (pts[:, 1] - hub[1]) ** 2
        tip_idx = int(np.argmax(d2))
        tip = (float(pts[tip_idx, 0]), float(pts[tip_idx, 1]))
        angle = math.degrees(math.atan2(tip[1] - hub[1], tip[0] - hub[0]))
        length = math.sqrt(float(d2[tip_idx]))
        refined.append(BladeObs(
            centroid=seed.centroid,
            angle_deg=angle,
            tip=tip,
            length=length,
            area=seed.area,
        ))
    return FramePose(frame_id=-1, hub=hub, blades=refined)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-dir", required=True,
                        help="directory of JPEG frames (use extract_frames.sh first)")
    parser.add_argument("--prompts", required=True,
                        help="JSON from pick_prompts.py")
    parser.add_argument("--checkpoint", required=True,
                        help="SAM2 checkpoint .pt (e.g. sam2.1_hiera_large.pt)")
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml",
                        help="SAM2 model config name (resolved via sam2's hydra config dir)")
    parser.add_argument("--output-dir", default="blade_tracker_sam2_output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-area", type=int, default=500,
                        help="reject masks below this pixel count")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="limit propagation to this many frames (0 = all)")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument("--no-overlay", dest="save_overlay", action="store_false")
    parser.add_argument("--hub", default=None, metavar="X,Y",
                        help="fixed hub position in full-res pixels. REQUIRED when "
                             "tracking only one blade. Optional with 2+ blades (overrides "
                             "the axis-intersection estimate).")
    parser.add_argument("--symmetry", type=int, default=0, metavar="N",
                        help="if >1, synthesize N-fold rotational symmetry: write extra "
                             "blade columns at the tracked angle + k*(360/N) degrees. "
                             "Use --symmetry 3 with a single tracked blade to fill out "
                             "all three rotor blades from the one measurement.")
    parser.add_argument("--gpu-frames", action="store_true",
                        help="keep all loaded frames on GPU (faster, but needs ~12 GB per "
                             "1000 frames at 1024-px input). Default: CPU offload — uses "
                             "host RAM, only the active batch goes to GPU.")
    parser.set_defaults(save_overlay=True)
    args = parser.parse_args()

    fixed_hub: tuple[float, float] | None = None
    if args.hub:
        xs, ys = args.hub.split(",")
        fixed_hub = (float(xs), float(ys))

    frame_dir = Path(args.frame_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = out_dir / "masks" if args.save_masks else None
    if mask_dir is not None:
        mask_dir.mkdir(exist_ok=True)

    frames = list_frames(frame_dir)
    if args.max_frames > 0:
        frames = frames[:args.max_frames]
    total = len(frames)
    print(f"frames: {total}")

    prompts = load_prompts(Path(args.prompts))
    seed_frame_idx = int(prompts.get("frame_idx", 0))
    print(f"seed frame: {seed_frame_idx}  prompts: {len(prompts['prompts'])}")

    n_tracked = len(prompts["prompts"])
    if n_tracked == 1 and fixed_hub is None:
        raise SystemExit(
            "Tracking only one blade — you must pass --hub X,Y so the per-frame pose "
            "has a stable rotation centre (one blade axis has no unique intersection).")
    if args.symmetry > 1 and fixed_hub is None:
        raise SystemExit(
            "--symmetry requires --hub X,Y so synthesized blade tips can be placed "
            "around a fixed rotation centre.")

    # ── SAM2 ────────────────────────────────────────────────────────────────
    predictor = build_sam2(Path(args.checkpoint), args.model_cfg, args.device)

    # Tell SAM2 about the frames. SAM2 expects the frame dir directly.
    t0 = time.monotonic()
    offload = not args.gpu_frames
    print(f"init_state ...  (offload_video_to_cpu={offload})")
    inference_state = predictor.init_state(
        video_path=str(frame_dir),
        offload_video_to_cpu=offload,
        offload_state_to_cpu=offload,
    )
    print(f"  init_state took {time.monotonic() - t0:.1f}s")

    for p in prompts["prompts"]:
        obj_id = int(p["obj_id"])
        call_kwargs: dict = dict(
            inference_state=inference_state,
            frame_idx=seed_frame_idx,
            obj_id=obj_id,
        )

        # Box prompt (strongest) ----------------------------------------
        if "box" in p:
            box = np.array(p["box"], dtype=np.float32)  # [xmin,ymin,xmax,ymax]
            call_kwargs["box"] = box

        # Multi-point prompts (v2 schema) -------------------------------
        if "points" in p:
            pts = np.array(p["points"], dtype=np.float32)
            labels = np.array(p.get("labels", [1] * len(pts)), dtype=np.int32)
            call_kwargs["points"] = pts
            call_kwargs["labels"] = labels

        # Legacy single-point schema ------------------------------------
        if "point" in p and "points" not in p:
            pt = np.array([p["point"]], dtype=np.float32)
            lab = np.array([int(p.get("label", 1))], dtype=np.int32)
            call_kwargs["points"] = pt
            call_kwargs["labels"] = lab

        if "box" not in call_kwargs and "points" not in call_kwargs:
            raise SystemExit(f"prompt for obj_id={obj_id} has no point/points/box")

        predictor.add_new_points_or_box(**call_kwargs)
        desc = []
        if "box" in call_kwargs:
            desc.append(f"box={p['box']}")
        if "points" in call_kwargs:
            desc.append(f"points={call_kwargs['points'].tolist()}")
        print(f"  added prompt: obj_id={obj_id}  {' '.join(desc)}")

    # ── Propagate + write outputs ───────────────────────────────────────────
    csv_path = out_dir / "poses.csv"
    csv_f = open(csv_path, "w", newline="")
    writer = csv.writer(csv_f)
    n_blades_out = max(n_tracked, args.symmetry) if args.symmetry > 1 else n_tracked
    header = ["frame_id", "pose_valid", "hub_x", "hub_y"]
    for k in range(n_blades_out):
        header += [f"b{k}_angle_deg", f"b{k}_tip_x", f"b{k}_tip_y", f"b{k}_length", f"b{k}_area"]
    writer.writerow(header)

    video_writer = None
    progress = Progress(total)
    n_locked = 0

    print("propagate_in_video ...")
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
        if frame_idx >= total:
            break
        progress.update(frame_idx)

        masks_per_obj: dict[int, np.ndarray] = {}
        for i, obj_id in enumerate(obj_ids):
            m = (mask_logits[i] > 0.0).cpu().numpy()
            if m.ndim == 3:
                m = m[0]
            masks_per_obj[int(obj_id)] = m.astype(np.uint8)

        pose = extract_pose_from_masks(masks_per_obj, args.min_area, fixed_hub=fixed_hub)

        if mask_dir is not None:
            combo = np.zeros_like(next(iter(masks_per_obj.values())), dtype=np.uint8)
            for k, (_oid, m) in enumerate(sorted(masks_per_obj.items())):
                combo[m > 0] = 80 + 60 * k
            cv2.imwrite(str(mask_dir / f"mask_{frame_idx:05d}.png"), combo)

        # Optional: synthesize symmetric blades from the tracked reference.
        if pose is not None and args.symmetry > 1 and len(pose.blades) >= 1:
            ref = pose.blades[0]
            existing = len(pose.blades)
            for k in range(existing, args.symmetry):
                step_deg = 360.0 / args.symmetry * k
                new_angle = ref.angle_deg + step_deg
                rad = math.radians(new_angle)
                tip = (pose.hub[0] + math.cos(rad) * ref.length,
                       pose.hub[1] + math.sin(rad) * ref.length)
                pose.blades.append(BladeObs(
                    centroid=pose.hub,
                    angle_deg=new_angle,
                    tip=tip,
                    length=ref.length,
                    area=0,  # synthetic, no real mask
                ))

        row: list = [frame_idx]
        if pose is not None:
            pose.frame_id = frame_idx
            n_locked += 1
            row += [1, f"{pose.hub[0]:.3f}", f"{pose.hub[1]:.3f}"]
            for k in range(n_blades_out):
                if k < len(pose.blades):
                    b = pose.blades[k]
                    row += [f"{b.angle_deg:.4f}", f"{b.tip[0]:.3f}",
                            f"{b.tip[1]:.3f}", f"{b.length:.3f}", b.area]
                else:
                    row += ["", "", "", "", ""]
        else:
            row += [0, "", ""] + [""] * (n_blades_out * 5)
        writer.writerow(row)

        if args.save_overlay:
            frame_bgr = cv2.imread(str(frames[frame_idx]))
            if pose is not None:
                pose.frame_id = frame_idx
                vis = draw_overlay(frame_bgr, pose)
            else:
                vis = frame_bgr
            if video_writer is None:
                h, w = vis.shape[:2]
                # avc1 (H.264) handles long 4K streams; mp4v fails past ~3700 4K frames
                video_writer = cv2.VideoWriter(
                    str(out_dir / "overlay.mp4"),
                    cv2.VideoWriter_fourcc(*"avc1"),
                    args.fps, (w, h),
                )
                if not video_writer.isOpened():
                    # avc1 not available in this OpenCV build — fall back to mp4v
                    video_writer = cv2.VideoWriter(
                        str(out_dir / "overlay.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        args.fps, (w, h),
                    )
            video_writer.write(vis)

    csv_f.close()
    if video_writer is not None:
        video_writer.release()
    progress.finish()

    print(f"locked: {n_locked} / {total}  ({100.0 * n_locked / max(1, total):.1f}%)")
    print(f"wrote {csv_path}")
    if args.save_overlay:
        print(f"wrote {out_dir / 'overlay.mp4'}")


if __name__ == "__main__":
    main()
