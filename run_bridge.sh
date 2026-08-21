#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

source /opt/ros/noetic/setup.bash

if [ -f /home/mobile/catkin_ws/devel/setup.bash ]; then
  source /home/mobile/catkin_ws/devel/setup.bash
fi

if [ -f /home/mobile/ranger_ws/devel/setup.bash ]; then
  source /home/mobile/ranger_ws/devel/setup.bash
fi

source /home/mobile/miniconda3/etc/profile.d/conda.sh
conda activate falcon

ACTION_TOPIC="${ACTION_TOPIC:-/falcon/action_id}"
DEPTH_TOPIC="${DEPTH_TOPIC:-/camera/depth/image_rect_raw}"
MAX_POLAR_AGE_SEC="${MAX_POLAR_AGE_SEC:-0.50}"
DATA_TIMEOUT_SEC="${DATA_TIMEOUT_SEC:-0.60}"
POLAR_BUFFER_SIZE="${POLAR_BUFFER_SIZE:-200}"

echo "Depth topic:               $DEPTH_TOPIC"
echo "Polar topic:               /tag_polar"
echo "Max depth-polar time diff: $MAX_POLAR_AGE_SEC s"
echo "Falcon input timeout:      $DATA_TIMEOUT_SEC s"
echo "Polar buffer size:         $POLAR_BUFFER_SIZE"

exec python "$REPO_ROOT/sensor/falcon_ros_bridge.py" \
  --checkpoint "$REPO_ROOT/weights/falcon_bc_70traj_action_head_lstm0045.pth" \
  --depth_topic "$DEPTH_TOPIC" \
  --polar_topic /tag_polar \
  --action_topic "$ACTION_TOPIC" \
  --max_polar_age_sec "$MAX_POLAR_AGE_SEC" \
  --data_timeout_sec "$DATA_TIMEOUT_SEC" \
  --polar_buffer_size "$POLAR_BUFFER_SIZE" \
  --debug_timing
