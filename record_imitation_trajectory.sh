#!/usr/bin/env bash
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
