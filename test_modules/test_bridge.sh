#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

python "$REPO_ROOT/sensor/falcon_ros_bridge.py" \
  --checkpoint "$REPO_ROOT/falcon_pretrained_25.pth" \
  --input_type depth \
  --polar_source topic \
  --depth_topic /camera/aligned_depth_to_color/image_raw \
  --polar_topic /tag_polar \
  --cmd_vel_topic /cmd_vel
