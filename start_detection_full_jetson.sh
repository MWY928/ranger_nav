#!/usr/bin/env bash
set -euo pipefail

# Jetson-local AprilTag + odometry-compensated polar tracker launcher.
# This replaces start_detection_jetson.sh; do not run both at the same time.

ROS_ENV_SCRIPT="${ROS_ENV_SCRIPT:-$HOME/go2_test_scripts/source_ros_jetson.sh}"
DETECTION_WS="${DETECTION_WS:-$HOME/go2_detection_ws}"
DETECTION_SETUP="${DETECTION_SETUP:-$DETECTION_WS/devel/setup.bash}"

GO2_IMAGE_TOPIC="${GO2_IMAGE_TOPIC:-/camera/color/image_raw}"
GO2_CAMERA_INFO_TOPIC="${GO2_CAMERA_INFO_TOPIC:-/camera/color/camera_info}"
TAG_DETECTIONS_TOPIC="${TAG_DETECTIONS_TOPIC:-/tag_detections}"
POLAR_TOPIC="${POLAR_TOPIC:-/tag_polar}"
TRACKING_STATE_TOPIC="${TRACKING_STATE_TOPIC:-/tag_tracking_state}"
ODOM_TOPIC="${ODOM_TOPIC:-/go2/sport_odom}"
OUTPUT_FRAME_ID="${OUTPUT_FRAME_ID:-base_link}"

TARGET_TAG_ID="${TARGET_TAG_ID:-0}"
USE_FIRST_DETECTION="${USE_FIRST_DETECTION:-false}"
THETA_OFFSET_RAD="${THETA_OFFSET_RAD:-0.0}"
ENABLE_THETA_OFFSET="${ENABLE_THETA_OFFSET:-true}"
THETA_DEADBAND_RAD="${THETA_DEADBAND_RAD:-0.0}"
ENABLE_THETA_DEADBAND="${ENABLE_THETA_DEADBAND:-true}"
CAMERA_OFFSET_X_M="${CAMERA_OFFSET_X_M:-0.0}"
CAMERA_OFFSET_Y_M="${CAMERA_OFFSET_Y_M:-0.0}"
DISTANCE_OFFSET="${DISTANCE_OFFSET:-0.6}"
MIN_DISTANCE="${MIN_DISTANCE:-0.0}"

LOST_TIMEOUT_SEC="${LOST_TIMEOUT_SEC:-0.12}"
PREDICT_TIMEOUT_SEC="${PREDICT_TIMEOUT_SEC:-6.0}"
PREDICT_RATE_HZ="${PREDICT_RATE_HZ:-15.0}"
REACQUIRE_RESET_SEC="${REACQUIRE_RESET_SEC:-1.0}"
DETECTION_STREAM_TIMEOUT_SEC="${DETECTION_STREAM_TIMEOUT_SEC:-0.5}"
TAG_SEARCH_ENABLED="${TAG_SEARCH_ENABLED:-true}"
TAG_SEARCH_TIMEOUT_SEC="${TAG_SEARCH_TIMEOUT_SEC:-12.0}"

ODOM_FILTER_ALPHA="${ODOM_FILTER_ALPHA:-0.5}"
ODOM_TIMEOUT_SEC="${ODOM_TIMEOUT_SEC:-0.25}"
MAX_ODOM_JUMP_M="${MAX_ODOM_JUMP_M:-0.75}"
MAX_ODOM_YAW_JUMP_RAD="${MAX_ODOM_YAW_JUMP_RAD:-1.20}"

if [[ ! -f "$ROS_ENV_SCRIPT" ]]; then
  echo "ERROR: ROS network setup script not found: $ROS_ENV_SCRIPT" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$ROS_ENV_SCRIPT"

if [[ ! -f /opt/ros/noetic/setup.bash ]]; then
  echo "ERROR: ROS Noetic setup not found." >&2
  exit 1
fi
# shellcheck disable=SC1091
source /opt/ros/noetic/setup.bash

if [[ ! -f "$DETECTION_SETUP" ]]; then
  echo "ERROR: Detection workspace has not been built: $DETECTION_SETUP" >&2
  echo "Build it first: cd $DETECTION_WS && catkin_make --pkg go_nav -DPYTHON_EXECUTABLE=/usr/bin/python3" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$DETECTION_SETUP"

if [[ -z "${ROS_MASTER_URI:-}" || -z "${ROS_IP:-}" ]]; then
  echo "ERROR: ROS_MASTER_URI or ROS_IP is not set by $ROS_ENV_SCRIPT" >&2
  exit 1
fi

for package_name in go_nav apriltag_ros; do
  if ! rospack find "$package_name" >/dev/null 2>&1; then
    echo "ERROR: ROS package '$package_name' is not visible." >&2
    exit 1
  fi
done

# The full launch owns both nodes. Refuse to create duplicate detections or
# duplicate /tag_polar publishers if the old launcher is still running.
running_nodes="$(rosnode list 2>/dev/null || true)"
for node_name in /apriltag_detector /tag_to_polar_node /polar_goal_tracker_node; do
  if grep -Fxq "$node_name" <<<"$running_nodes"; then
    echo "ERROR: ROS node $node_name is already running." >&2
    echo "Stop the old detection launcher before starting this full launcher." >&2
    exit 1
  fi
done

check_topic_type() {
  local topic_name="$1"
  local expected_type="$2"
  local actual_type
  if ! actual_type="$(rostopic type "$topic_name" 2>/dev/null)"; then
    echo "ERROR: Cannot find required topic: $topic_name" >&2
    exit 1
  fi
  if [[ "$actual_type" != "$expected_type" ]]; then
    echo "ERROR: $topic_name has type '$actual_type', expected '$expected_type'." >&2
    exit 1
  fi
}

check_topic_type "$GO2_IMAGE_TOPIC" sensor_msgs/Image
check_topic_type "$GO2_CAMERA_INFO_TOPIC" sensor_msgs/CameraInfo
check_topic_type "$ODOM_TOPIC" nav_msgs/Odometry

echo "ROS master:                 $ROS_MASTER_URI"
echo "Jetson ROS IP:              $ROS_IP"
echo "Image / CameraInfo:         $GO2_IMAGE_TOPIC / $GO2_CAMERA_INFO_TOPIC"
echo "Detections / polar:         $TAG_DETECTIONS_TOPIC / $POLAR_TOPIC"
echo "Tracking state topic:       $TRACKING_STATE_TOPIC"
echo "Go2 odometry topic:         $ODOM_TOPIC"
echo "Prediction:                 $LOST_TIMEOUT_SEC-$PREDICT_TIMEOUT_SEC s at $PREDICT_RATE_HZ Hz"
echo "Search request:             enabled=$TAG_SEARCH_ENABLED timeout=$TAG_SEARCH_TIMEOUT_SEC s"
echo "Starting Jetson-local AprilTag + odometry tracker..."

exec roslaunch go_nav go2_detection_full.launch \
  image_topic:="$GO2_IMAGE_TOPIC" \
  camera_info_topic:="$GO2_CAMERA_INFO_TOPIC" \
  detections_topic:="$TAG_DETECTIONS_TOPIC" \
  polar_topic:="$POLAR_TOPIC" \
  tracking_state_topic:="$TRACKING_STATE_TOPIC" \
  odom_topic:="$ODOM_TOPIC" \
  output_frame_id:="$OUTPUT_FRAME_ID" \
  target_tag_id:="$TARGET_TAG_ID" \
  use_first_detection:="$USE_FIRST_DETECTION" \
  theta_offset_rad:="$THETA_OFFSET_RAD" \
  enable_theta_offset:="$ENABLE_THETA_OFFSET" \
  theta_deadband_rad:="$THETA_DEADBAND_RAD" \
  enable_theta_deadband:="$ENABLE_THETA_DEADBAND" \
  camera_offset_x_m:="$CAMERA_OFFSET_X_M" \
  camera_offset_y_m:="$CAMERA_OFFSET_Y_M" \
  distance_offset:="$DISTANCE_OFFSET" \
  min_distance:="$MIN_DISTANCE" \
  use_odom_fallback:=true \
  lost_timeout_sec:="$LOST_TIMEOUT_SEC" \
  predict_timeout_sec:="$PREDICT_TIMEOUT_SEC" \
  publish_rate_hz:="$PREDICT_RATE_HZ" \
  reacquire_reset_sec:="$REACQUIRE_RESET_SEC" \
  detection_stream_timeout_sec:="$DETECTION_STREAM_TIMEOUT_SEC" \
  search_enabled:="$TAG_SEARCH_ENABLED" \
  search_timeout_sec:="$TAG_SEARCH_TIMEOUT_SEC" \
  alpha:="$ODOM_FILTER_ALPHA" \
  odom_timeout_sec:="$ODOM_TIMEOUT_SEC" \
  max_odom_jump_m:="$MAX_ODOM_JUMP_M" \
  max_odom_yaw_jump_rad:="$MAX_ODOM_YAW_JUMP_RAD"
