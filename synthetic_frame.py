#!/usr/bin/env python3
"""
Synthetic dataset generator for vibration stabilisation testing.

Modes:
  aruco  — 5 ArUco markers (IDs 1-4 corner anchors + ID 0 centre vibrating)
            For use with aruco_homography.cpp
  ransac — ID 0 only at centre (background speckle for ORB+RANSAC)
            For use with ransac_stabilizer.cpp

Usage:
    python synthetic_frame.py --mode aruco
    python synthetic_frame.py --mode ransac --output synthetic_data_no_aruco
    python synthetic_frame.py --mode aruco --frames 500 --output my_dataset
"""

import numpy as np
import cv2
import json
import os
from pathlib import Path
import argparse

# ── Marker configuration ─────────────────────────────────────────────────────
ANCHOR_IDS    = [1, 2, 3, 4]
ANCHOR_SIZE   = 120
ANCHOR_MARGIN = 50
TARGET_SIZE   = 150      # ID-0 measurement marker
TARGET_MARGIN = 30       # exclusion zone around target for RANSAC features


class SyntheticDataGenerator:

    def __init__(self, width=1440, height=1080):
        self.width  = width
        self.height = height

    # ── Texture ───────────────────────────────────────────────────────────────

    def create_base_texture(self):
        """
        Spray-paint speckle pattern.
        Two overlapping coats of fine dots on a light-grey base,
        mimicking real black-on-white DIC speckle patterns.
        seed=42 → identical background every run (motion is the only diff).
        """
        rng = np.random.default_rng(seed=42)
        texture = np.full((self.height, self.width), 200, dtype=np.float32)

        # Primary coat
        n = 100_000
        xs       = rng.integers(0, self.width,  size=n)
        ys       = rng.integers(0, self.height, size=n)
        opac     = rng.beta(0.6, 2.0, size=n)
        dot_vals = (200 * (1 - opac)).astype(np.float32)
        radii    = rng.choice([1, 1, 1, 2, 2, 3], size=n)
        for i in range(n):
            cv2.circle(texture, (int(xs[i]), int(ys[i])),
                       int(radii[i]), float(dot_vals[i]), -1)

        # Second coat (slightly denser fine dots — natural spray variation)
        n2  = 40_000
        xs2 = rng.integers(0, self.width,  size=n2)
        ys2 = rng.integers(0, self.height, size=n2)
        for i in range(n2):
            cv2.circle(texture, (int(xs2[i]), int(ys2[i])),
                       1, float(200 * (1 - rng.beta(0.4, 2.5))), -1)

        # Overspray halos (bright dots from mist bouncing off surface)
        n3  = 20_000
        xs3 = rng.integers(0, self.width,  size=n3)
        ys3 = rng.integers(0, self.height, size=n3)
        for i in range(n3):
            cv2.circle(texture, (int(xs3[i]), int(ys3[i])), 1, 230.0, -1)

        # Soft edges — real spray dots bleed slightly into surrounding paint
        texture = cv2.GaussianBlur(texture, (3, 3), 0.8)
        texture = np.clip(texture, 0, 255).astype(np.uint8)
        return cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)

    # ── Markers ───────────────────────────────────────────────────────────────

    def add_anchor_markers(self, image):
        """Stamp 4 corner ArUco markers (IDs 1-4). Used in aruco mode only."""
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        m, sz = ANCHOR_MARGIN, ANCHOR_SIZE
        positions = {
            1: (m,              m              ),  # TL
            2: (self.width-sz-m, m             ),  # TR
            3: (self.width-sz-m, self.height-sz-m),# BR
            4: (m,              self.height-sz-m), # BL
        }
        for mid, (x, y) in positions.items():
            marker = cv2.aruco.generateImageMarker(aruco_dict, mid, sz)
            marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
            image[y:y+sz, x:x+sz] = marker_bgr
        return image

    def add_measurement_marker(self, image, struct_dx=0.0, struct_dy=0.0):
        """Stamp ArUco marker ID 0 at frame centre, offset by structural vibration."""
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        marker_size = TARGET_SIZE

        cx = int(self.width  // 2 - marker_size // 2 + struct_dx)
        cy = int(self.height // 2 - marker_size // 2 + struct_dy)

        if not (0 <= cx and 0 <= cy and
                cx + marker_size <= self.width and cy + marker_size <= self.height):
            return image

        marker     = cv2.aruco.generateImageMarker(aruco_dict, 0, marker_size)
        marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        image[cy:cy + marker_size, cx:cx + marker_size] = marker_bgr
        return image

    # ── Motion generators ─────────────────────────────────────────────────────

    def generate_structural_vibration(self, frame_idx, fps=100):
        t = frame_idx / fps
        frequencies = [15.0, 45.0, 85.0]
        amplitudes  = [0.05, 0.15, 0.25]
        phases      = [0.0,  1.57, 3.14]
        dx = dy = 0.0
        for freq, amp, phase in zip(frequencies, amplitudes, phases):
            dx += amp * np.sin(2 * np.pi * freq * t + phase)
            dy += amp * np.cos(2 * np.pi * freq * t + phase + np.pi / 4)
        return dx, dy

    def generate_camera_sway(self, frame_idx, fps=100):
        if not hasattr(self, '_sway_state'):
            self._sway_state = {'x': 0.0, 'y': 0.0}
        t = frame_idx / fps

        base_x = 8.0 * np.sin(2 * np.pi * 0.15 * t)
        base_y = 6.0 * np.cos(2 * np.pi * 0.12 * t)

        self._sway_state['x'] = self._sway_state['x'] * 0.90 + np.random.normal(0, 2.0)
        self._sway_state['y'] = self._sway_state['y'] * 0.90 + np.random.normal(0, 2.0)

        dx = np.clip(base_x + self._sway_state['x'], -20.0, 20.0)
        dy = np.clip(base_y + self._sway_state['y'], -20.0, 20.0)
        return dx, dy

    def generate_rotation(self, frame_idx, fps=100):
        t = frame_idx / fps
        angle_deg = 0.5 * np.sin(2 * np.pi * 0.8 * t)
        cx, cy = self.width // 2, self.height // 2
        R = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        return angle_deg, R

    def apply_sensor_noise(self, image):
        image_float = image.astype(np.float32)
        image_float += np.random.normal(0, 0.5, image.shape)
        poisson_input = np.maximum(image_float * 0.1, 0.1)
        image_float += np.random.poisson(poisson_input) / 0.1 - image_float
        return np.clip(np.round(image_float), 0, 255).astype(np.uint8)

    # ── Dataset generation ────────────────────────────────────────────────────

    def generate_dataset(self, mode="aruco", num_frames=300,
                         output_dir=None):
        if output_dir is None:
            output_dir = "synthetic_data_aruco" if mode == "aruco" else "synthetic_data_ransac"

        Path(output_dir).mkdir(exist_ok=True)
        frames_dir = Path(output_dir) / "frames"
        frames_dir.mkdir(exist_ok=True)

        # Build base texture
        base_texture = self.create_base_texture()
        if mode == "aruco":
            base_texture = self.add_anchor_markers(base_texture)

        # Ground truth structure
        ground_truth = {
            "structural_vibration": {"displacement_x": [], "displacement_y": []},
            "camera_sway":          {"displacement_x": [], "displacement_y": []},
            "rotation":             {"angles": []},
        }

        if mode == "aruco":
            ground_truth["aruco_marker_id0"] = {"corners": [], "center": []}
        else:
            ground_truth["metadata"] = {
                "target_size_px": TARGET_SIZE,
                "target_centre":  [self.width // 2, self.height // 2],
                "fps":            100,
                "description":    "No anchor ArUco markers. Target patch vibrates "
                                  "at centre. Background speckle moves with camera only."
            }

        print(f"Mode: {mode}")
        print(f"Generating {num_frames} frames → {output_dir}/")

        marker_size = TARGET_SIZE

        for frame_idx in range(num_frames):
            if frame_idx % 50 == 0:
                print(f"  Frame {frame_idx}/{num_frames}")

            struct_dx, struct_dy = self.generate_structural_vibration(frame_idx)
            sway_dx,   sway_dy  = self.generate_camera_sway(frame_idx)
            angle_deg, R        = self.generate_rotation(frame_idx)

            # Step 1: fresh copy of static background
            current_frame = base_texture.copy()

            # Step 2: stamp measurement marker (ID 0) with structural vibration
            current_frame = self.add_measurement_marker(
                current_frame, struct_dx, struct_dy)

            # ── Compute ArUco ID 0 image-space coordinates (aruco mode) ──
            if mode == "aruco":
                center_x = self.width  // 2 - marker_size // 2
                center_y = self.height // 2 - marker_size // 2
                aruco_x = center_x + struct_dx
                aruco_y = center_y + struct_dy
                corners_before = [
                    [aruco_x, aruco_y],
                    [aruco_x + marker_size, aruco_y],
                    [aruco_x + marker_size, aruco_y + marker_size],
                    [aruco_x, aruco_y + marker_size],
                ]
                # Apply sway + rotation to each corner
                final_corners = []
                for corner in corners_before:
                    tx_c = corner[0] + sway_dx
                    ty_c = corner[1] + sway_dy
                    if abs(angle_deg) > 0.001:
                        cx0, cy0 = self.width // 2, self.height // 2
                        dx0 = tx_c - cx0
                        dy0 = ty_c - cy0
                        angle_rad = np.radians(angle_deg)
                        cos_a = np.cos(angle_rad)
                        sin_a = np.sin(angle_rad)
                        tx_c = dx0 * cos_a - dy0 * sin_a + cx0
                        ty_c = dx0 * sin_a + dy0 * cos_a + cy0
                    final_corners.append([tx_c, ty_c])
                final_center = [
                    sum(c[0] for c in final_corners) / 4,
                    sum(c[1] for c in final_corners) / 4,
                ]
                ground_truth["aruco_marker_id0"]["corners"].append(final_corners)
                ground_truth["aruco_marker_id0"]["center"].append(final_center)

            # Step 3: camera sway (whole frame)
            if abs(sway_dx) > 0.01 or abs(sway_dy) > 0.01:
                M_sway = np.float32([[1, 0, sway_dx], [0, 1, sway_dy]])
                current_frame = cv2.warpAffine(
                    current_frame, M_sway, (self.width, self.height))

            # Step 4: camera rotation
            if abs(angle_deg) > 0.001:
                current_frame = cv2.warpAffine(
                    current_frame, R, (self.width, self.height))

            # Step 5: sensor noise
            current_frame = self.apply_sensor_noise(current_frame)

            cv2.imwrite(str(frames_dir / f"frame_{frame_idx:04d}.png"),
                        current_frame)

            # Store ground truth
            ground_truth["structural_vibration"]["displacement_x"].append(struct_dx)
            ground_truth["structural_vibration"]["displacement_y"].append(struct_dy)
            ground_truth["camera_sway"]["displacement_x"].append(sway_dx)
            ground_truth["camera_sway"]["displacement_y"].append(sway_dy)
            ground_truth["rotation"]["angles"].append(angle_deg)

        with open(Path(output_dir) / "ground_truth.json", 'w') as f:
            json.dump(ground_truth, f, indent=2)

        print(f"Dataset generated: {output_dir}")
        return output_dir


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic dataset")
    parser.add_argument("--mode", type=str, default="aruco",
                        choices=["aruco", "ransac"],
                        help="aruco: 5 markers (4 anchors + ID 0). "
                             "ransac: ID 0 only, ORB+RANSAC on background.")
    parser.add_argument("--frames", type=int, default=300,
                        help="Number of frames")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: auto per mode)")
    args = parser.parse_args()

    generator = SyntheticDataGenerator()
    generator.generate_dataset(args.mode, args.frames, args.output)


if __name__ == "__main__":
    main()
