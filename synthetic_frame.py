#!/usr/bin/env python3

import numpy as np
import cv2
import json
import os
from pathlib import Path
import argparse

class SyntheticDataGenerator:
    
    def __init__(self, width=1440, height=1080):
        self.width = width
        self.height = height
        
    def create_base_texture(self):
        """checkerboard pattern """
        texture = np.zeros((self.height, self.width), dtype=np.uint8)
        square_size = 40
        
        for y in range(0, self.height, square_size):
            for x in range(0, self.width, square_size):
                if (x // square_size + y // square_size) % 2 == 0:
                    texture[y:y+square_size, x:x+square_size] = 220
                else:
                    texture[y:y+square_size, x:x+square_size] = 40
        
        return cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
    
    def add_aruco_markers(self, image):

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        marker_size = 100
        
        positions = [(200, 200), (1200, 200), (200, 800), (1200, 800)]
        
        for i, (x, y) in enumerate(positions):
            marker = cv2.aruco.generateImageMarker(aruco_dict, i, marker_size)
            marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
            
            if x + marker_size < self.width and y + marker_size < self.height:
                image[y:y+marker_size, x:x+marker_size] = marker_bgr
        
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
    
    def generate_dataset(self, num_frames=300, output_dir="synthetic_data"):
        Path(output_dir).mkdir(exist_ok=True)
        frames_dir = Path(output_dir) / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        base_texture = self.create_base_texture()
        base_texture = self.add_aruco_markers(base_texture)
        
        ground_truth = {
            "structural_vibration": {
                "displacement_x": [],
                "displacement_y": []
            }
        }
        
        print(f"Generating {num_frames} frames...")
        
        for frame_idx in range(num_frames):
            if frame_idx % 50 == 0:
                print(f"  Frame {frame_idx}/{num_frames}")
            
            dx, dy = self.generate_structural_vibration(frame_idx)
            
            current_frame = base_texture.copy()
            if abs(dx) > 0.01 or abs(dy) > 0.01:
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                current_frame = cv2.warpAffine(current_frame, M, (self.width, self.height))
            
            frame_path = frames_dir / f"frame_{frame_idx:04d}.png"
            cv2.imwrite(str(frame_path), current_frame)
            
            ground_truth["structural_vibration"]["displacement_x"].append(dx)
            ground_truth["structural_vibration"]["displacement_y"].append(dy)
        
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
