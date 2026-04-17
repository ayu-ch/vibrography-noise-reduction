#!/usr/bin/env python3
"""
Extract real RANSAC noise characteristics from compensation.csv files.
Outputs statistics that can be fed into stabilizer_env.py for realistic training.

Usage:
    python -m rl_stabilizer.training.analyze_real_data compensation.csv
    python -m rl_stabilizer.training.analyze_real_data *.csv
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def analyze(csv_path):
    df = pd.read_csv(csv_path)
    print(f"\n{'='*60}")
    print(f"File: {csv_path} ({len(df)} frames)")
    print(f"{'='*60}")

    # Inlier distribution
    print(f"\nInlier distribution:")
    print(f"  mean: {df.inliers.mean():.0f}")
    print(f"  std:  {df.inliers.std():.0f}")
    print(f"  min:  {df.inliers.min()}")
    print(f"  max:  {df.inliers.max()}")

    # Frame-to-frame jitter = RANSAC noise
    dtx = df.tx_px.diff().dropna()
    dty = df.ty_px.diff().dropna()
    print(f"\nFrame-to-frame jitter (RANSAC noise):")
    print(f"  tx std:      {dtx.std():.4f} px")
    print(f"  tx kurtosis: {dtx.kurtosis():.2f}  (Gaussian=0, heavy-tailed>0)")
    print(f"  ty std:      {dty.std():.4f} px")
    print(f"  ty kurtosis: {dty.kurtosis():.2f}")

    # Displacement range (= camera sway)
    print(f"\nCamera sway range:")
    print(f"  tx: [{df.tx_px.min():.2f}, {df.tx_px.max():.2f}] (total {df.tx_px.max()-df.tx_px.min():.1f} px)")
    print(f"  ty: [{df.ty_px.min():.2f}, {df.ty_px.max():.2f}] (total {df.ty_px.max()-df.ty_px.min():.1f} px)")

    # Rotation and scale
    print(f"\nRotation std: {df.rotation_deg.std():.6f} deg")
    print(f"Scale std:    {(df.scale - 1.0).std():.6f}")

    # Dropout rate
    invalid = len(df[df.homography_valid == 0])
    print(f"\nH failures: {invalid}/{len(df)} ({100*invalid/len(df):.1f}%)")

    # Jitter vs inliers correlation
    jitter = np.sqrt(dtx**2 + dty.values**2)
    inl = df.inliers.iloc[1:]
    bins = pd.qcut(inl, 4, labels=['low', 'med-low', 'med-high', 'high'], duplicates='drop')
    print(f"\nJitter by inlier quartile:")
    for label in ['low', 'med-low', 'med-high', 'high']:
        mask = bins == label
        if mask.sum() > 0:
            print(f"  {label:>8}: jitter={jitter[mask].mean():.4f} px, inliers={inl[mask].mean():.0f}")

    # Output env parameters
    print(f"\n{'='*60}")
    print("Suggested stabilizer_env.py updates:")
    print(f"{'='*60}")
    print(f"  ransac_inlier_mean = {df.inliers.mean():.0f}")
    print(f"  ransac_inlier_std  = {df.inliers.std():.0f}")
    print(f"  ransac_noise_sigma = {dtx.std():.2f}  # tx jitter std")
    print(f"  sway_amp_x range   = [0, {df.tx_px.max():.0f}]  # from displacement range")
    print(f"  sway_amp_y range   = [0, {max(abs(df.ty_px.min()), df.ty_px.max()):.0f}]")
    print(f"  dropout_rate       = {invalid/len(df):.3f}")
    print(f"  kurtosis_tx        = {dtx.kurtosis():.2f}  # >0 means use heavy-tailed noise")

    return {
        'inlier_mean': df.inliers.mean(),
        'inlier_std': df.inliers.std(),
        'noise_sigma_tx': dtx.std(),
        'noise_sigma_ty': dty.std(),
        'kurtosis_tx': dtx.kurtosis(),
        'kurtosis_ty': dty.kurtosis(),
        'sway_max_tx': df.tx_px.max() - df.tx_px.min(),
        'sway_max_ty': df.ty_px.max() - df.ty_px.min(),
        'dropout_rate': invalid / len(df),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csvs", nargs="+", help="compensation.csv files")
    args = parser.parse_args()

    all_stats = []
    for csv in args.csvs:
        stats = analyze(csv)
        all_stats.append(stats)

    if len(all_stats) > 1:
        print(f"\n{'='*60}")
        print(f"Aggregate across {len(all_stats)} videos:")
        print(f"{'='*60}")
        for key in all_stats[0]:
            vals = [s[key] for s in all_stats]
            print(f"  {key}: mean={np.mean(vals):.3f}, range=[{np.min(vals):.3f}, {np.max(vals):.3f}]")


if __name__ == "__main__":
    main()
