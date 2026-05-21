#!/usr/bin/env bash
# Extract a video into JPEG frames for SAM2 video predictor.
#
# Usage:
#   ./extract_frames.sh <video.mp4> <out_dir> [stride] [scale] [max_frames]
#
#   stride      default 1   (every Nth frame)
#   scale       default 1.0 (resize factor, e.g. 0.5 for half-res)
#   max_frames  default 0   (0 = all; otherwise stop after this many frames)
#
# SAM2 expects frames named 00000.jpg, 00001.jpg, ...
set -euo pipefail

VIDEO=${1:?video path}
OUT=${2:?output dir}
STRIDE=${3:-1}
SCALE=${4:-1.0}
MAX_FRAMES=${5:-0}

mkdir -p "$OUT"

VF="select='not(mod(n,${STRIDE}))'"
if [ "$SCALE" != "1.0" ]; then
    VF="${VF},scale=iw*${SCALE}:ih*${SCALE}"
fi

FRAME_LIMIT=""
if [ "$MAX_FRAMES" != "0" ]; then
    FRAME_LIMIT="-frames:v $MAX_FRAMES"
fi

echo "extracting frames: stride=$STRIDE scale=$SCALE max_frames=$MAX_FRAMES -> $OUT"
ffmpeg -hide_banner -loglevel warning -stats -i "$VIDEO" \
    -vf "$VF" -vsync vfr -q:v 2 $FRAME_LIMIT \
    "$OUT/%05d.jpg"

N=$(ls "$OUT" | wc -l)
echo "wrote $N frames to $OUT"
