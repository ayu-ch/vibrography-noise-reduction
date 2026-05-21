"""
Stage 1 — classical wind-turbine blade tracker.

Per frame, recover:
  - hub centre (x, y) in pixels
  - per-blade angle (deg, CCW from +x) and tip (x, y)

Pipeline:
  1. Threshold the bright + low-saturation region  -> candidate blade mask
  2. Drop the sky component (connected to the top edge of the frame)
  3. Connected components, take the three largest by area
  4. PCA principal axis per component -> seed angle
  5. Estimate hub as least-squares intersection of the three blade axes
  6. Re-resolve each blade angle/tip as farthest-mask-pixel-from-hub
  7. Associate IDs across frames by angular proximity to previous frame

Outputs:
  <out_dir>/poses.csv          per-frame pose
  <out_dir>/overlay.mp4        visualisation (unless --no-overlay)
  <out_dir>/mask_<i>.png       debug masks (only with --save-masks)
"""

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class BladeObs:
    centroid: tuple[float, float]
    angle_deg: float
    tip: tuple[float, float]
    length: float
    area: int


@dataclass
class FramePose:
    frame_id: int
    hub: tuple[float, float]
    blades: list[BladeObs]


def segment_blade_mask(bgr: np.ndarray, v_thresh: int, s_thresh: int) -> np.ndarray:
    """Bright + low-saturation pixels = white blade or white sky."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    mask = ((v >= v_thresh) & (s <= s_thresh)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def drop_sky(mask: np.ndarray, labels: np.ndarray, stats: np.ndarray) -> set[int]:
    """Return label indices that touch the top edge — treat as sky."""
    top_row_labels = set(np.unique(labels[0, :]).tolist())
    top_row_labels.discard(0)
    return top_row_labels


def pca_axis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    mean = points.mean(axis=0)
    centred = points - mean
    cov = np.cov(centred.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, -1]
    angle = math.degrees(math.atan2(axis[1], axis[0]))
    proj = centred @ axis
    half_len = float(proj.max() - proj.min()) * 0.5
    return mean, axis, angle, half_len


def intersect_axes(blades: list[BladeObs]) -> tuple[float, float]:
    """Least-squares intersection of lines (each line = centroid + t * axis)."""
    A = np.zeros((2, 2))
    b = np.zeros(2)
    for blade in blades:
        theta = math.radians(blade.angle_deg)
        axis = np.array([math.cos(theta), math.sin(theta)])
        n = np.array([-axis[1], axis[0]])
        nnT = np.outer(n, n)
        A += nnT
        b += nnT @ np.array(blade.centroid)
    try:
        hub = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        hub = np.mean([blade.centroid for blade in blades], axis=0)
    return float(hub[0]), float(hub[1])


def process_frame(
    bgr: np.ndarray,
    frame_id: int,
    v_thresh: int,
    s_thresh: int,
    min_area: int,
    max_blades: int = 3,
    debug_mask_dir: Path | None = None,
) -> tuple[FramePose | None, np.ndarray]:
    mask = segment_blade_mask(bgr, v_thresh, s_thresh)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sky_labels = drop_sky(mask, labels, stats)

    candidates = []
    for i in range(1, num):
        if i in sky_labels:
            continue
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        candidates.append(i)
    candidates.sort(key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
    candidates = candidates[:max_blades]

    blade_mask = np.zeros_like(mask)
    for i in candidates:
        blade_mask[labels == i] = 255

    if debug_mask_dir is not None:
        cv2.imwrite(str(debug_mask_dir / f"mask_{frame_id:05d}.png"), blade_mask)

    if not candidates:
        return None, blade_mask

    seeds: list[BladeObs] = []
    point_sets: list[np.ndarray] = []
    for label_idx in candidates:
        ys, xs = np.where(labels == label_idx)
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        mean, axis, angle, half_len = pca_axis(pts)
        seeds.append(BladeObs(
            centroid=(float(mean[0]), float(mean[1])),
            angle_deg=angle,
            tip=(float(mean[0] + axis[0] * half_len), float(mean[1] + axis[1] * half_len)),
            length=half_len * 2.0,
            area=int(stats[label_idx, cv2.CC_STAT_AREA]),
        ))
        point_sets.append(pts)

    hub = intersect_axes(seeds) if len(seeds) >= 2 else seeds[0].centroid

    refined: list[BladeObs] = []
    for seed, pts in zip(seeds, point_sets):
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

    return FramePose(frame_id=frame_id, hub=hub, blades=refined), blade_mask


def angle_diff_deg(a: float, b: float) -> float:
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def assign_ids(pose: FramePose, prev: FramePose | None) -> list[BladeObs]:
    """Return blades reordered so blade[k] this frame matches blade[k] last frame."""
    if prev is None or not prev.blades:
        return sorted(pose.blades, key=lambda b: b.angle_deg)

    n_prev = len(prev.blades)
    n_now = len(pose.blades)
    cost = np.full((n_prev, n_now), 1e6)
    for i, pb in enumerate(prev.blades):
        for j, cb in enumerate(pose.blades):
            cost[i, j] = angle_diff_deg(pb.angle_deg, cb.angle_deg)

    assigned = [None] * n_prev
    used = set()
    for i in range(n_prev):
        order = np.argsort(cost[i])
        for j in order:
            if j in used:
                continue
            if cost[i, j] > 60.0:
                break
            assigned[i] = pose.blades[j]
            used.add(j)
            break

    extras = [b for j, b in enumerate(pose.blades) if j not in used]
    out = [b for b in assigned if b is not None] + extras
    return out


def draw_overlay(bgr: np.ndarray, pose: FramePose) -> np.ndarray:
    out = bgr.copy()
    hub = (int(round(pose.hub[0])), int(round(pose.hub[1])))
    cv2.circle(out, hub, 10, (0, 0, 255), 2)
    cv2.drawMarker(out, hub, (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
    palette = [(0, 255, 0), (255, 255, 0), (255, 0, 255), (255, 128, 0)]
    for i, b in enumerate(pose.blades):
        color = palette[i % len(palette)]
        tip = (int(round(b.tip[0])), int(round(b.tip[1])))
        cv2.line(out, hub, tip, color, 2)
        cv2.circle(out, tip, 6, color, 2)
        label = f"B{i} {b.angle_deg:+.1f} deg"
        cv2.putText(out, label, (tip[0] + 8, tip[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.putText(out, f"frame {pose.frame_id}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def count_frames(path: Path) -> int | None:
    """Best-effort total frame count for progress reporting. None if unknown."""
    if path.is_dir():
        return sum(1 for p in path.iterdir()
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    suffix = path.suffix.lower()
    if suffix in {".mp4", ".mov", ".avi", ".mkv"}:
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n if n > 0 else None
    if suffix in {".png", ".jpg", ".jpeg", ".bmp"}:
        return 1
    return None


class Progress:
    """In-place progress line: [ 1234 / 9000 ]  13.7%  86.2 fps  ETA 0:01:30."""
    def __init__(self, total: int | None, every: int = 30):
        self.total = total
        self.every = max(1, every)
        self.t0 = time.monotonic()
        self.last_print = 0.0
        self.last_i = 0

    def update(self, i: int) -> None:
        now = time.monotonic()
        if i - self.last_i < self.every and (now - self.last_print) < 0.5:
            return
        elapsed = max(1e-3, now - self.t0)
        fps = (i + 1) / elapsed
        if self.total:
            pct = 100.0 * (i + 1) / self.total
            remaining = max(0.0, (self.total - i - 1) / max(1e-3, fps))
            eta = time.strftime("%H:%M:%S", time.gmtime(remaining))
            line = f"\r[ {i+1:>6d} / {self.total:>6d} ]  {pct:5.1f}%  {fps:6.1f} fps  ETA {eta}"
        else:
            line = f"\r[ {i+1:>6d} ]  {fps:6.1f} fps"
        sys.stdout.write(line)
        sys.stdout.flush()
        self.last_print = now
        self.last_i = i

    def finish(self) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()


def open_source(path: Path):
    """Yield (frame_id, bgr) regardless of whether path is a video, folder, or image."""
    if path.is_dir():
        files = sorted([p for p in path.iterdir()
                        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])
        for i, p in enumerate(files):
            img = cv2.imread(str(p))
            if img is not None:
                yield i, img
        return
    suffix = path.suffix.lower()
    if suffix in {".mp4", ".mov", ".avi", ".mkv"}:
        cap = cv2.VideoCapture(str(path))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield i, frame
            i += 1
        cap.release()
        return
    img = cv2.imread(str(path))
    if img is not None:
        yield 0, img


def tune_mode(path: Path, v_thresh: int, s_thresh: int, min_area: int) -> None:
    """Interactive trackbars over the first frame to dial in thresholds."""
    gen = open_source(path)
    try:
        _, frame = next(gen)
    except StopIteration:
        print(f"No frames in {path}")
        return

    win = "blade_tracker tuning"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("V_thresh", win, v_thresh, 255, lambda _v: None)
    cv2.createTrackbar("S_thresh", win, s_thresh, 255, lambda _v: None)
    cv2.createTrackbar("min_area/100", win, max(1, min_area // 100), 200, lambda _v: None)

    print("Adjust trackbars. Press 's' to print current values, 'q' to quit.")
    while True:
        v = cv2.getTrackbarPos("V_thresh", win)
        s = cv2.getTrackbarPos("S_thresh", win)
        a = cv2.getTrackbarPos("min_area/100", win) * 100
        pose, mask = process_frame(frame, 0, v, s, max(50, a))
        vis = frame.copy()
        vis = cv2.addWeighted(vis, 0.6,
                              cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.4, 0)
        if pose is not None:
            vis = draw_overlay(vis, pose)
        cv2.imshow(win, vis)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            print(f"--v-thresh {v} --s-thresh {s} --min-area {max(50, a)}")
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="video file, image, or folder of images")
    parser.add_argument("--output-dir", default="blade_tracker_output")
    parser.add_argument("--v-thresh", type=int, default=200,
                        help="HSV value lower bound (bright pixels)")
    parser.add_argument("--s-thresh", type=int, default=40,
                        help="HSV saturation upper bound (near-white pixels)")
    parser.add_argument("--min-area", type=int, default=2000,
                        help="reject blade components smaller than this (px)")
    parser.add_argument("--max-blades", type=int, default=3)
    parser.add_argument("--no-overlay", dest="save_overlay", action="store_false")
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="stop after this many frames (0 = process all)")
    parser.add_argument("--tune", action="store_true",
                        help="interactive trackbar tuning on the first frame")
    parser.set_defaults(save_overlay=True)
    args = parser.parse_args()

    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"input not found: {src}")

    if args.tune:
        tune_mode(src, args.v_thresh, args.s_thresh, args.min_area)
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = out_dir / "masks" if args.save_masks else None
    if mask_dir is not None:
        mask_dir.mkdir(exist_ok=True)

    csv_path = out_dir / "poses.csv"
    csv_f = open(csv_path, "w", newline="")
    writer = csv.writer(csv_f)
    header = ["frame_id", "hub_x", "hub_y"]
    for k in range(args.max_blades):
        header += [f"b{k}_angle_deg", f"b{k}_tip_x", f"b{k}_tip_y", f"b{k}_length"]
    writer.writerow(header)

    video_writer = None
    prev: FramePose | None = None
    n_seen = 0
    n_locked = 0
    total = count_frames(src)
    if args.max_frames > 0:
        total = min(total, args.max_frames) if total else args.max_frames
    progress = Progress(total)

    for frame_id, frame in open_source(src):
        if args.max_frames > 0 and frame_id >= args.max_frames:
            break
        n_seen += 1
        progress.update(frame_id)
        pose, _mask = process_frame(
            frame, frame_id,
            args.v_thresh, args.s_thresh, args.min_area,
            max_blades=args.max_blades,
            debug_mask_dir=mask_dir,
        )

        row: list = [frame_id]
        if pose is not None:
            pose.blades = assign_ids(pose, prev)
            prev = pose
            n_locked += 1
            row += [f"{pose.hub[0]:.3f}", f"{pose.hub[1]:.3f}"]
            for k in range(args.max_blades):
                if k < len(pose.blades):
                    b = pose.blades[k]
                    row += [f"{b.angle_deg:.4f}", f"{b.tip[0]:.3f}", f"{b.tip[1]:.3f}", f"{b.length:.3f}"]
                else:
                    row += ["", "", "", ""]
        else:
            row += ["", ""] + [""] * (args.max_blades * 4)
        writer.writerow(row)

        if args.save_overlay:
            vis = draw_overlay(frame, pose) if pose is not None else frame
            if video_writer is None:
                h, w = vis.shape[:2]
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

    print(f"frames seen: {n_seen}  locked: {n_locked}  ({100.0 * n_locked / max(1, n_seen):.1f}%)")
    print(f"wrote {csv_path}")
    if args.save_overlay:
        print(f"wrote {out_dir / 'overlay.mp4'}")


if __name__ == "__main__":
    main()
