#!/usr/bin/env python3

import numpy as np
import cv2
import json
import os
from pathlib import Path
import argparse

ANCHOR_IDS    = [1, 2, 3, 4]
ANCHOR_SIZE   = 120
ANCHOR_MARGIN = 50    

class SyntheticDataGenerator:
    
    def __init__(self, width=1440, height=1080):
        self.width = width
        self.height = height
        
    def create_base_texture(self):
        """Low-contrast checkerboard"""
        texture = np.ones((self.height, self.width), dtype=np.uint8) * 180
        square_size = 60
        
        # Very low contrast checkerboard (won't interfere with ArUco)
        for y in range(0, self.height, square_size):
            for x in range(0, self.width, square_size):
                if (x // square_size + y // square_size) % 2 == 0:
                    texture[y:y+square_size, x:x+square_size] = 190
                else:
                    texture[y:y+square_size, x:x+square_size] = 170
        
        return cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
    
    def add_aruco_markers(self, image, struct_dx=0, struct_dy=0):
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        marker_size = 150
        
        center_x = self.width // 2 - marker_size // 2
        center_y = self.height // 2 - marker_size // 2
        
        x = int(center_x + struct_dx)
        y = int(center_y + struct_dy)
        
        marker = cv2.aruco.generateImageMarker(aruco_dict, 0, marker_size)
        marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        
        if (x >= 0 and y >= 0 and 
            x + marker_size < self.width and y + marker_size < self.height):
            image[y:y+marker_size, x:x+marker_size] = marker_bgr
        
        return image
    
    def add_anchor_markers(self, image):
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

    
    def generate_structural_vibration(self, frame_idx, fps=100):
        t = frame_idx / fps
        
        # Multi-frequency vibration components
        frequencies = [15.0, 45.0, 85.0]  # Hz
        amplitudes = [0.05, 0.15, 0.25]   # pixels
        phases = [0.0, 1.57, 3.14]        # radians
        
        dx = dy = 0.0
        for freq, amp, phase in zip(frequencies, amplitudes, phases):
            dx += amp * np.sin(2 * np.pi * freq * t + phase)
            dy += amp * np.cos(2 * np.pi * freq * t + phase + np.pi/4)
        
        return dx, dy
    
    def generate_camera_sway(self, frame_idx, fps=100):
        """Generate camera sway: low-frequency random walk [several to tens of px, large amplitude]."""
        if not hasattr(self, '_sway_state'):
            self._sway_state = {'x': 0.0, 'y': 0.0}
        
        t = frame_idx / fps
        
        # Low-frequency sinusoidal base with random walk overlay
        base_freq_x = 0.15  # Hz - low frequency
        base_freq_y = 0.12  # Hz - low frequency  
        base_amp_x = 8.0    # pixels - several px base
        base_amp_y = 6.0    # pixels - several px base
        
        base_x = base_amp_x * np.sin(2 * np.pi * base_freq_x * t)
        base_y = base_amp_y * np.cos(2 * np.pi * base_freq_y * t)
        
        # Random walk component to reach "tens of px"
        random_factor = 2.0
        walk_step_x = np.random.normal(0, random_factor)
        walk_step_y = np.random.normal(0, random_factor)
        
        # Update random walk state with decay
        decay = 0.90
        self._sway_state['x'] = self._sway_state['x'] * decay + walk_step_x
        self._sway_state['y'] = self._sway_state['y'] * decay + walk_step_y
        
        # Combine base motion and random walk
        dx = base_x + self._sway_state['x']
        dy = base_y + self._sway_state['y']
        
        # Clamp to "tens of px" range (several to ~20px)
        max_displacement = 20.0 
        dx = np.clip(dx, -max_displacement, max_displacement)
        dy = np.clip(dy, -max_displacement, max_displacement)
        
        return dx, dy
    
    def generate_rotation(self, frame_idx, fps=100):
        """Generate camera rotation: ±0.5° random rotation."""
        t = frame_idx / fps
        
        # Rotation parameters
        max_angle = 0.5  # degrees (as per specification)
        frequency = 0.8  # Hz
        
        angle_deg = max_angle * np.sin(2 * np.pi * frequency * t)
        
        center_x, center_y = self.width // 2, self.height // 2
        rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle_deg, 1.0)
        
        return angle_deg, rotation_matrix
    
    def apply_sensor_noise(self, image):
        """Apply Gaussian noise"""
        image_float = image.astype(np.float32)
        
        # Gaussian noise
        gaussian_sigma = 0.5
        gaussian_noise = np.random.normal(0, gaussian_sigma, image.shape)
        image_float += gaussian_noise
        
        # Shot noise (Poisson-like)
        shot_factor = 0.1
        poisson_input = np.maximum(image_float * shot_factor, 0.1)
        shot_noise = np.random.poisson(poisson_input) / shot_factor - image_float
        image_float += shot_noise
        
        # Quantization (8-bit)
        image_float = np.round(image_float * 255.0 / 255.0)
        
        return np.clip(image_float, 0, 255).astype(np.uint8)
    
    def generate_dataset(self, num_frames=300, output_dir="synthetic_data"):
        Path(output_dir).mkdir(exist_ok=True)
        frames_dir = Path(output_dir) / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        base_texture = self.create_base_texture()
        base_texture = self.add_anchor_markers(base_texture)
        
        ground_truth = {
            "structural_vibration": {
                "displacement_x": [],
                "displacement_y": []
            },
            "camera_sway": {
                "displacement_x": [],
                "displacement_y": []
            },
            "rotation": {
                "angles": []
            },
            "aruco_marker_id0": {
                "corners": [],  # 4 corners: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                "center": []    # center point: [cx, cy]
            }
        }
        
        print(f"Generating {num_frames} frames...")
        
        for frame_idx in range(num_frames):
            if frame_idx % 50 == 0:
                print(f"  Frame {frame_idx}/{num_frames}")
            
            struct_dx, struct_dy = self.generate_structural_vibration(frame_idx)
            sway_dx, sway_dy = self.generate_camera_sway(frame_idx)
            angle_deg, rotation_matrix = self.generate_rotation(frame_idx)
            
            # Create base texture (stationary background)
            current_frame = base_texture.copy()
            
            # Add ArUco markers with structural vibration (markers move, background doesn't)
            current_frame = self.add_aruco_markers(current_frame, struct_dx, struct_dy)
            
            # Calculate ArUco marker ID 0 coordinates BEFORE camera motion
            marker_size = 150
            center_x = self.width // 2 - marker_size // 2
            center_y = self.height // 2 - marker_size // 2
            
            # ArUco marker corners (before camera motion)
            aruco_x = center_x + struct_dx
            aruco_y = center_y + struct_dy
            corners_before_camera = [
                [aruco_x, aruco_y],                           # Top-left
                [aruco_x + marker_size, aruco_y],             # Top-right
                [aruco_x + marker_size, aruco_y + marker_size], # Bottom-right
                [aruco_x, aruco_y + marker_size]              # Bottom-left
            ]
            center_before_camera = [aruco_x + marker_size/2, aruco_y + marker_size/2]
            
            # Apply camera motion transformations to coordinates
            final_corners = []
            for corner in corners_before_camera:
                # Apply camera sway
                transformed_corner = [corner[0] + sway_dx, corner[1] + sway_dy]
                
                # Apply rotation around image center
                if abs(angle_deg) > 0.001:
                    cx, cy = self.width // 2, self.height // 2
                    # Translate to origin
                    tx = transformed_corner[0] - cx
                    ty = transformed_corner[1] - cy
                    # Rotate
                    angle_rad = np.radians(angle_deg)
                    cos_a = np.cos(angle_rad)
                    sin_a = np.sin(angle_rad)
                    rx = tx * cos_a - ty * sin_a
                    ry = tx * sin_a + ty * cos_a
                    # Translate back
                    transformed_corner = [rx + cx, ry + cy]
                
                final_corners.append(transformed_corner)
            
            # Calculate final center
            final_center = [
                sum(corner[0] for corner in final_corners) / 4,
                sum(corner[1] for corner in final_corners) / 4
            ]
            
            # Camera sway
            if abs(sway_dx) > 0.01 or abs(sway_dy) > 0.01:
                M_sway = np.float32([[1, 0, sway_dx], [0, 1, sway_dy]])
                current_frame = cv2.warpAffine(current_frame, M_sway, (self.width, self.height))
            
            # Apply rotation
            if abs(angle_deg) > 0.001:
                current_frame = cv2.warpAffine(current_frame, rotation_matrix, (self.width, self.height))
            
            # Apply sensor noise
            current_frame = self.apply_sensor_noise(current_frame)
            
            frame_path = frames_dir / f"frame_{frame_idx:04d}.png"
            cv2.imwrite(str(frame_path), current_frame)
            
            # Store ground truth
            ground_truth["structural_vibration"]["displacement_x"].append(struct_dx)
            ground_truth["structural_vibration"]["displacement_y"].append(struct_dy)
            ground_truth["camera_sway"]["displacement_x"].append(sway_dx)
            ground_truth["camera_sway"]["displacement_y"].append(sway_dy)
            ground_truth["rotation"]["angles"].append(angle_deg)
            ground_truth["aruco_marker_id0"]["corners"].append(final_corners)
            ground_truth["aruco_marker_id0"]["center"].append(final_center)
        
        with open(Path(output_dir) / "ground_truth.json", 'w') as f:
            json.dump(ground_truth, f, indent=2)
        
        print(f"Dataset generated: {output_dir}")
        return output_dir

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic dataset")
    parser.add_argument("--frames", type=int, default=300, help="Number of frames")
    parser.add_argument("--output", type=str, default="synthetic_data", help="Output directory")
    
    args = parser.parse_args()
    
    generator = SyntheticDataGenerator()
    generator.generate_dataset(args.frames, args.output)

if __name__ == "__main__":
    main()
