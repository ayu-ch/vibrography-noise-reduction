"""
Pick one prompt point per blade on a reference frame.

Used to seed SAM2 video propagation: each click defines an object that SAM2 will
track across the whole video. Three clicks -> three blade IDs.

Output JSON:
    {
        "video": "<path>",
        "frame_idx": 0,
        "prompts": [
            {"obj_id": 1, "point": [x, y], "label": 1},
            {"obj_id": 2, "point": [x, y], "label": 1},
            {"obj_id": 3, "point": [x, y], "label": 1}
        ]
    }
"""

import argparse
import json
from pathlib import Path

import cv2


def grab_frame(path: Path, frame_idx: int):
    suffix = path.suffix.lower()
    if path.is_dir():
        files = sorted([p for p in path.iterdir()
                        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])
        if frame_idx >= len(files):
            raise SystemExit(f"frame_idx {frame_idx} out of range ({len(files)} files)")
        return cv2.imread(str(files[frame_idx]))
    if suffix in {".mp4", ".mov", ".avi", ".mkv"}:
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit(f"could not read frame {frame_idx} from {path}")
        return frame
    return cv2.imread(str(path))


def parse_points(specs: list[str]) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for s in specs:
        x_str, y_str = s.split(",")
        pts.append((int(x_str), int(y_str)))
    return pts


def parse_blade_points(spec: str) -> list[tuple[int, int]]:
    """Parse 'x1,y1 x2,y2 ...' OR 'x1,y1;x2,y2;...' -> [(x1,y1),(x2,y2),...]."""
    parts = spec.replace(";", " ").split()
    return parse_points(parts)


def parse_box(spec: str) -> tuple[int, int, int, int]:
    """Parse 'xmin,ymin,xmax,ymax' -> (xmin,ymin,xmax,ymax)."""
    nums = [int(x) for x in spec.replace(",", " ").split()]
    if len(nums) != 4:
        raise SystemExit(f"--blade-box expects 4 ints, got {nums}")
    xmin, ymin, xmax, ymax = nums
    if xmax <= xmin or ymax <= ymin:
        raise SystemExit(f"box must have xmax>xmin and ymax>ymin: {nums}")
    return (xmin, ymin, xmax, ymax)


def save_prompts_legacy(out_path: Path, src: Path, frame_idx: int,
                        frame_size: tuple[int, int],
                        pts: list[tuple[int, int]]) -> None:
    """One single point per blade — original format."""
    payload = {
        "video": str(src),
        "frame_idx": frame_idx,
        "frame_size": list(frame_size),
        "prompts": [
            {"obj_id": i + 1, "point": [int(x), int(y)], "label": 1}
            for i, (x, y) in enumerate(pts)
        ],
    }
    _write(out_path, payload)


def save_prompts_multi(out_path: Path, src: Path, frame_idx: int,
                       frame_size: tuple[int, int],
                       blades: list[dict]) -> None:
    """
    blades: list, one entry per object, each is:
        {"points": [[x,y],...], "labels": [1,...]}  OR
        {"box": [xmin,ymin,xmax,ymax]}              OR both.
    """
    prompts = []
    for i, b in enumerate(blades):
        entry = {"obj_id": i + 1}
        if "points" in b:
            entry["points"] = [[int(x), int(y)] for x, y in b["points"]]
            entry["labels"] = [int(l) for l in b.get("labels", [1] * len(b["points"]))]
        if "box" in b:
            entry["box"] = [int(v) for v in b["box"]]
        prompts.append(entry)
    payload = {
        "video": str(src),
        "frame_idx": frame_idx,
        "frame_size": list(frame_size),
        "prompts": prompts,
    }
    _write(out_path, payload)


def _write(out_path: Path, payload: dict) -> None:
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}")
    print(json.dumps(payload, indent=2))


def dump_reference_frame(frame, out_path: Path) -> None:
    cv2.imwrite(str(out_path), frame)
    print(f"wrote reference frame: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="video file or frame folder")
    parser.add_argument("--frame-idx", type=int, default=0)
    parser.add_argument("--n-blades", type=int, default=3)
    parser.add_argument("--out", default="prompts.json")
    parser.add_argument("--display-scale", type=float, default=0.0,
                        help="scale factor for display window (0 = auto-fit to 1200 px wide)")
    parser.add_argument("--points", nargs="+", default=None,
                        metavar="X,Y",
                        help="bypass GUI: ONE 'x,y' per blade (single-point prompts, weakest)")
    parser.add_argument("--blade", action="append", default=None,
                        metavar="'x,y x,y ...'",
                        help="one --blade per object, each a space-separated list of points. "
                             "Stronger than --points. Example: --blade '450,702 510,900' --blade ...")
    parser.add_argument("--blade-box", action="append", default=None,
                        metavar="'xmin,ymin,xmax,ymax'",
                        help="one --blade-box per object: bounding rectangle around the blade. "
                             "Strongest prompt — recommended for featureless blades.")
    parser.add_argument("--dump-frame", default=None,
                        help="save the reference frame as PNG and exit (no clicking needed)")
    args = parser.parse_args()

    src = Path(args.input)
    frame = grab_frame(src, args.frame_idx)
    if frame is None:
        raise SystemExit(f"failed to load frame from {src}")
    h, w = frame.shape[:2]

    if args.dump_frame:
        dump_reference_frame(frame, Path(args.dump_frame))
        return

    if args.points:
        pts = parse_points(args.points)
        if len(pts) != args.n_blades:
            raise SystemExit(
                f"--points expects {args.n_blades} entries, got {len(pts)}"
            )
        for x, y in pts:
            if not (0 <= x < w and 0 <= y < h):
                raise SystemExit(f"point ({x},{y}) is outside frame {w}x{h}")
        save_prompts_legacy(Path(args.out), src, args.frame_idx, (w, h), pts)
        return

    if args.blade or args.blade_box:
        n_pt = len(args.blade or [])
        n_bx = len(args.blade_box or [])
        n_objs = max(n_pt, n_bx)
        if n_pt and n_bx and n_pt != n_bx:
            raise SystemExit(
                f"if you mix --blade and --blade-box, give the same number of each "
                f"(got {n_pt} and {n_bx})"
            )
        blades: list[dict] = []
        for i in range(n_objs):
            entry: dict = {}
            if args.blade and i < n_pt:
                pts_i = parse_blade_points(args.blade[i])
                for x, y in pts_i:
                    if not (0 <= x < w and 0 <= y < h):
                        raise SystemExit(f"point ({x},{y}) is outside frame {w}x{h}")
                entry["points"] = pts_i
                entry["labels"] = [1] * len(pts_i)
            if args.blade_box and i < n_bx:
                box = parse_box(args.blade_box[i])
                xmin, ymin, xmax, ymax = box
                if not (0 <= xmin < w and 0 <= xmax <= w
                        and 0 <= ymin < h and 0 <= ymax <= h):
                    raise SystemExit(f"box {box} is outside frame {w}x{h}")
                entry["box"] = list(box)
            blades.append(entry)
        save_prompts_multi(Path(args.out), src, args.frame_idx, (w, h), blades)
        return

    scale = args.display_scale
    if scale <= 0.0:
        scale = min(1.0, 1200.0 / w)
    disp_w = int(round(w * scale))
    disp_h = int(round(h * scale))
    disp = cv2.resize(frame, (disp_w, disp_h)) if scale != 1.0 else frame.copy()

    clicks: list[tuple[int, int]] = []
    palette = [(0, 255, 0), (255, 255, 0), (255, 0, 255), (255, 128, 0), (0, 200, 255)]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < args.n_blades:
            # store in full-resolution coords
            fx = int(round(x / scale))
            fy = int(round(y / scale))
            clicks.append((fx, fy))

    win = "pick_prompts — click ONE point on each blade, then 's' to save, 'r' to reset, 'q' to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    cv2.resizeWindow(win, disp_w, disp_h)

    print(f"frame size: {w}x{h}, display scale: {scale:.3f}")
    print(f"click {args.n_blades} points (one per blade). 's' save, 'r' reset, 'q' quit.")

    while True:
        vis = disp.copy()
        for i, (fx, fy) in enumerate(clicks):
            x = int(round(fx * scale))
            y = int(round(fy * scale))
            color = palette[i % len(palette)]
            cv2.circle(vis, (x, y), 8, color, 2)
            cv2.putText(vis, f"B{i}", (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow(win, vis)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            print("quit without saving")
            cv2.destroyAllWindows()
            return
        if key == ord('r'):
            clicks.clear()
            print("reset")
        if key == ord('s'):
            if len(clicks) != args.n_blades:
                print(f"need {args.n_blades} clicks, got {len(clicks)}")
                continue
            break

    cv2.destroyAllWindows()
    save_prompts_legacy(Path(args.out), src, args.frame_idx, (w, h), clicks)


if __name__ == "__main__":
    main()
