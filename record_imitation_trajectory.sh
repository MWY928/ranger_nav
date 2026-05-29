#!/usr/bin/env bash

# =========================================================
# record_imitation_trajectory.sh
#
# Purpose:
#   Start the imitation-learning trajectory collector outside a ROS workspace.
#
# This records synchronized samples from:
#   1. /tag_polar:
#      goal as PointStamped, where point.x = r and point.y = theta
#   2. /camera/aligned_depth_to_color/image_raw:
#      depth processed to 256x256 float32 meters, clipped to max 10m
#   3. discrete action labels:
#      0 stop, 1 forward, 2 left, 3 right
#
# Default behavior:
#   - Runs in heuristic mode, so this node publishes /cmd_vel itself.
#   - Uses forward speed = 0.3 and turn speed = 0.3 from the Python defaults.
#   - Saves data under:
#       test_modules/test_results/il_trajectories/<RUN_ID>
#
# Examples:
#   bash record_imitation_trajectory.sh --sample_limit 200
#   bash record_imitation_trajectory.sh --control_mode passive --action_source_topic /cmd_vel
#
# Extra arguments are forwarded to sensor/collect_imitation_trajectory.py.
# =========================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

if [ -f /opt/ros/noetic/setup.bash ]; then
  source /opt/ros/noetic/setup.bash
fi
if [ -f /home/mobile/catkin_ws/devel/setup.bash ]; then
  source /home/mobile/catkin_ws/devel/setup.bash
fi
if [ -f /home/mobile/ranger_ws/devel/setup.bash ]; then
  source /home/mobile/ranger_ws/devel/setup.bash
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="$REPO_ROOT/test_modules/test_results/il_trajectories"

mkdir -p "$OUTPUT_DIR"

echo "Recording imitation trajectory to: $OUTPUT_DIR/$RUN_ID"

exec python "$REPO_ROOT/sensor/collect_imitation_trajectory.py" \
  --depth_topic /camera/aligned_depth_to_color/image_raw \
  --polar_topic /tag_polar \
  --cmd_vel_topic /cmd_vel \
  --control_mode heuristic \
  --output_dir "$OUTPUT_DIR" \
  --run_id "$RUN_ID" \
  "$@"
