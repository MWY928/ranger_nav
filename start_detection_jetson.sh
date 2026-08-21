#!/usr/bin/env bash
set -euo pipefail

# Jetson-local AprilTag detection launcher.
#
# Expected layout:
#   ROS network setup:  ~/go2_test_scripts/source_ros_jetson.sh
#   Detection workspace: ~/go2_detection_ws
#
# All paths and detection parameters can be overridden with environment
# variables before running this script.

ROS_ENV_SCRIPT="${ROS_ENV_SCRIPT:-$HOME/go2_test_scripts/source_ros_jetson.sh}"
DETECTION_WS="${DETECTION_WS:-$HOME/go2_detection_ws}"
DETECTION_SETUP="${DETECTION_SETUP:-$DETECTION_WS/devel/setup.bash}"

GO2_IMAGE_TOPIC="${GO2_IMAGE_TOPIC:-/camera/color/image_raw}"
GO2_CAMERA_INFO_TOPIC="${GO2_CAMERA_INFO_TOPIC:-/camera/color/camera_info}"
TAG_DETECTIONS_TOPIC="${TAG_DETECTIONS_TOPIC:-/tag_detections}"
POLAR_TOPIC="${POLAR_TOPIC:-/tag_polar}"
OUTPUT_FRAME_ID="${OUTPUT_FRAME_ID:-base_link}"
TARGET_TAG_ID="${TARGET_TAG_ID:-0}"
USE_FIRST_DETECTION="${USE_FIRST_DETECTION:-false}"
THETA_OFFSET_RAD="${THETA_OFFSET_RAD:-0.0}"
ENABLE_THETA_OFFSET="${ENABLE_THETA_OFFSET:-true}"
THETA_DEADBAND_RAD="${THETA_DEADBAND_RAD:-0.0}"
ENABLE_THETA_DEADBAND="${ENABLE_THETA_DEADBAND:-true}"
DISTANCE_OFFSET="${DISTANCE_OFFSET:-0.6}"
MIN_DISTANCE="${MIN_DISTANCE:-0.0}"

if [[ -f "$ROS_ENV_SCRIPT" ]]; then
  # shellcheck disable=SC1090
  source "$ROS_ENV_SCRIPT"
else
  echo "ERROR: ROS network setup script not found: $ROS_ENV_SCRIPT" >&2
  echo "Set ROS_ENV_SCRIPT to the correct path, or create the file first." >&2
  exit 1
fi

# The ROS environment script normally sources these already. Source them here
# as a fallback so this launcher also works with a minimal IP-only setup file.
if [[ -f /opt/ros/noetic/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/noetic/setup.bash
else
  echo "ERROR: ROS Noetic setup not found: /opt/ros/noetic/setup.bash" >&2
  exit 1
fi

if [[ -f "$DETECTION_SETUP" ]]; then
  # shellcheck disable=SC1090
  source "$DETECTION_SETUP"
else
  echo "ERROR: Detection workspace has not been built: $DETECTION_SETUP" >&2
  echo "Build it first with:" >&2
  echo "  cd $DETECTION_WS && catkin_make" >&2
  exit 1
fi

if [[ -z "${ROS_MASTER_URI:-}" || -z "${ROS_IP:-}" ]]; then
  echo "ERROR: ROS_MASTER_URI or ROS_IP is not set by $ROS_ENV_SCRIPT" >&2
  exit 1
fi

if ! rospack find go_nav >/dev/null 2>&1; then
  echo "ERROR: ROS package 'go_nav' is not visible after sourcing $DETECTION_SETUP" >&2
  exit 1
fi

if ! rospack find apriltag_ros >/dev/null 2>&1; then
  echo "ERROR: ROS package 'apriltag_ros' is not installed." >&2
  echo "Install it with: sudo apt install ros-noetic-apriltag-ros" >&2
  exit 1
fi

echo "ROS master:                 $ROS_MASTER_URI"
echo "Jetson ROS IP:              $ROS_IP"
echo "Image topic:                $GO2_IMAGE_TOPIC"
echo "CameraInfo topic:           $GO2_CAMERA_INFO_TOPIC"
echo "AprilTag detections topic:  $TAG_DETECTIONS_TOPIC"
echo "Polar output topic:         $POLAR_TOPIC"
echo "Target tag ID:              $TARGET_TAG_ID"
echo "Distance offset:            $DISTANCE_OFFSET m"

if ! image_type="$(rostopic type "$GO2_IMAGE_TOPIC" 2>/dev/null)"; then
  echo "ERROR: Cannot find image topic: $GO2_IMAGE_TOPIC" >&2
  echo "Make sure roscore and the RealSense node are running." >&2
  exit 1
fi

if [[ "$image_type" != "sensor_msgs/Image" ]]; then
  echo "ERROR: $GO2_IMAGE_TOPIC has type '$image_type', expected 'sensor_msgs/Image'." >&2
  exit 1
fi

if ! camera_info_type="$(rostopic type "$GO2_CAMERA_INFO_TOPIC" 2>/dev/null)"; then
  echo "ERROR: Cannot find CameraInfo topic: $GO2_CAMERA_INFO_TOPIC" >&2
  echo "Make sure the RealSense node publishes color CameraInfo." >&2
  exit 1
fi

if [[ "$camera_info_type" != "sensor_msgs/CameraInfo" ]]; then
  echo "ERROR: $GO2_CAMERA_INFO_TOPIC has type '$camera_info_type', expected 'sensor_msgs/CameraInfo'." >&2
  exit 1
fi

echo "Camera topics are available. Starting Jetson-local AprilTag detection..."

exec roslaunch go_nav go2_detection_simple.launch \
  image_topic:="$GO2_IMAGE_TOPIC" \
  camera_info_topic:="$GO2_CAMERA_INFO_TOPIC" \
  detections_topic:="$TAG_DETECTIONS_TOPIC" \
  polar_topic:="$POLAR_TOPIC" \
  output_frame_id:="$OUTPUT_FRAME_ID" \
  target_tag_id:="$TARGET_TAG_ID" \
  use_first_detection:="$USE_FIRST_DETECTION" \
  theta_offset_rad:="$THETA_OFFSET_RAD" \
  enable_theta_offset:="$ENABLE_THETA_OFFSET" \
  theta_deadband_rad:="$THETA_DEADBAND_RAD" \
  enable_theta_deadband:="$ENABLE_THETA_DEADBAND" \
  distance_offset:="$DISTANCE_OFFSET" \
  min_distance:="$MIN_DISTANCE"
