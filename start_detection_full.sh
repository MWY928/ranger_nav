#!/bin/bash
set -e

source /opt/ros/noetic/setup.bash
source /home/mobile/catkin_ws/devel/setup.bash
source /home/mobile/ranger_ws/devel/setup.bash

source /home/mobile/miniconda3/etc/profile.d/conda.sh
conda activate simple_nav

GO2_IMAGE_TOPIC="${GO2_IMAGE_TOPIC:-/camera/color/image_raw}"
GO2_CAMERA_INFO_TOPIC="${GO2_CAMERA_INFO_TOPIC:-/camera/color/camera_info}"
TAG_DETECTIONS_TOPIC="${TAG_DETECTIONS_TOPIC:-/tag_detections}"
POLAR_TOPIC="${POLAR_TOPIC:-/tag_polar}"
TRACKING_STATE_TOPIC="${TRACKING_STATE_TOPIC:-/tag_tracking_state}"
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
USE_ODOM_FALLBACK="${USE_ODOM_FALLBACK:-true}"
ODOM_TOPIC="${ODOM_TOPIC:-/go2/sport_odom}"
LOST_TIMEOUT_SEC="${LOST_TIMEOUT_SEC:-0.12}"
PREDICT_TIMEOUT_SEC="${PREDICT_TIMEOUT_SEC:-6.0}"
PREDICT_RATE_HZ="${PREDICT_RATE_HZ:-15.0}"
REACQUIRE_RESET_SEC="${REACQUIRE_RESET_SEC:-1.0}"
DETECTION_STREAM_TIMEOUT_SEC="${DETECTION_STREAM_TIMEOUT_SEC:-0.5}"
TAG_SEARCH_ENABLED="${TAG_SEARCH_ENABLED:-false}"
TAG_SEARCH_TIMEOUT_SEC="${TAG_SEARCH_TIMEOUT_SEC:-12.0}"
ODOM_TIMEOUT_SEC="${ODOM_TIMEOUT_SEC:-0.25}"
ODOM_FILTER_ALPHA="${ODOM_FILTER_ALPHA:-0.5}"
MAX_ODOM_JUMP_M="${MAX_ODOM_JUMP_M:-0.75}"
MAX_ODOM_YAW_JUMP_RAD="${MAX_ODOM_YAW_JUMP_RAD:-1.20}"

echo "Go2 D435 image topic:       $GO2_IMAGE_TOPIC"
echo "Go2 D435 camera_info topic: $GO2_CAMERA_INFO_TOPIC"
echo "AprilTag detections topic:  $TAG_DETECTIONS_TOPIC"
echo "Polar output topic:         $POLAR_TOPIC"
echo "Tracking state topic:       $TRACKING_STATE_TOPIC"
echo "Camera offset in base:      x=$CAMERA_OFFSET_X_M y=$CAMERA_OFFSET_Y_M m"
echo "Use odom fallback:          $USE_ODOM_FALLBACK"
echo "Go2 odometry topic:         $ODOM_TOPIC"
echo "Prediction window:          $LOST_TIMEOUT_SEC-$PREDICT_TIMEOUT_SEC s"
echo "Search state:               enabled=$TAG_SEARCH_ENABLED timeout=$TAG_SEARCH_TIMEOUT_SEC s"

exec roslaunch go_nav go2_detection_full.launch \
  image_topic:="$GO2_IMAGE_TOPIC" \
  camera_info_topic:="$GO2_CAMERA_INFO_TOPIC" \
  detections_topic:="$TAG_DETECTIONS_TOPIC" \
  polar_topic:="$POLAR_TOPIC" \
  tracking_state_topic:="$TRACKING_STATE_TOPIC" \
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
  use_odom_fallback:="$USE_ODOM_FALLBACK" \
  odom_topic:="$ODOM_TOPIC" \
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
