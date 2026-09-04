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
DATA_TIMEOUT_SEC="${DATA_TIMEOUT_SEC:-0.90}"
POLAR_BUFFER_SIZE="${POLAR_BUFFER_SIZE:-200}"
ACTION_FILTER_ENABLED="${ACTION_FILTER_ENABLED:-true}"
ACTION_FILTER_TAU_SEC="${ACTION_FILTER_TAU_SEC:-0.15}"
ACTION_SWITCH_MARGIN="${ACTION_SWITCH_MARGIN:-0.10}"
ACTION_SWITCH_HOLD_SEC="${ACTION_SWITCH_HOLD_SEC:-0.12}"
STOP_SWITCH_HOLD_SEC="${STOP_SWITCH_HOLD_SEC:-0.20}"
TAG_SEARCH_ENABLED="${TAG_SEARCH_ENABLED:-false}"
TRACKING_STATE_TOPIC="${TRACKING_STATE_TOPIC:-/tag_tracking_state}"
TRACKING_STATE_TIMEOUT_SEC="${TRACKING_STATE_TIMEOUT_SEC:-0.50}"
TAG_SEARCH_TIMEOUT_SEC="${TAG_SEARCH_TIMEOUT_SEC:-12.0}"
TAG_SEARCH_DEFAULT_DIRECTION="${TAG_SEARCH_DEFAULT_DIRECTION:-left}"

echo "Depth topic:               $DEPTH_TOPIC"
echo "Polar topic:               /tag_polar"
echo "Max depth-polar time diff: $MAX_POLAR_AGE_SEC s"
echo "Falcon input timeout:      $DATA_TIMEOUT_SEC s"
echo "Polar buffer size:         $POLAR_BUFFER_SIZE"
echo "Action filter enabled:     $ACTION_FILTER_ENABLED"
echo "Action filter tau:         $ACTION_FILTER_TAU_SEC s"
echo "Action switch margin:      $ACTION_SWITCH_MARGIN"
echo "Action switch hold:        $ACTION_SWITCH_HOLD_SEC s"
echo "Stop switch hold:          $STOP_SWITCH_HOLD_SEC s"
echo "Tag search enabled:        $TAG_SEARCH_ENABLED"
echo "Tracking state topic:      $TRACKING_STATE_TOPIC"
echo "Tracking state timeout:    $TRACKING_STATE_TIMEOUT_SEC s"
echo "Tag search timeout:        $TAG_SEARCH_TIMEOUT_SEC s"
echo "Tag search fallback:       $TAG_SEARCH_DEFAULT_DIRECTION"

exec python "$REPO_ROOT/sensor/falcon_ros_bridge.py" \
  --checkpoint "$REPO_ROOT/weights/falcon_bc_70traj_action_head_lstm0045.pth" \
  --depth_topic "$DEPTH_TOPIC" \
  --polar_topic /tag_polar \
  --action_topic "$ACTION_TOPIC" \
  --max_polar_age_sec "$MAX_POLAR_AGE_SEC" \
  --data_timeout_sec "$DATA_TIMEOUT_SEC" \
  --polar_buffer_size "$POLAR_BUFFER_SIZE" \
  --deterministic \
  --action_filter_enabled "$ACTION_FILTER_ENABLED" \
  --action_filter_tau_sec "$ACTION_FILTER_TAU_SEC" \
  --action_switch_margin "$ACTION_SWITCH_MARGIN" \
  --action_switch_hold_sec "$ACTION_SWITCH_HOLD_SEC" \
  --stop_switch_hold_sec "$STOP_SWITCH_HOLD_SEC" \
  --tag_search_enabled "$TAG_SEARCH_ENABLED" \
  --tracking_state_topic "$TRACKING_STATE_TOPIC" \
  --tracking_state_timeout_sec "$TRACKING_STATE_TIMEOUT_SEC" \
  --tag_search_timeout_sec "$TAG_SEARCH_TIMEOUT_SEC" \
  --tag_search_default_direction "$TAG_SEARCH_DEFAULT_DIRECTION" \
  --debug_timing \
  --debug_mapping
