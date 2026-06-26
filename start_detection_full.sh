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
OUTPUT_FRAME_ID="${OUTPUT_FRAME_ID:-base_link}"
TARGET_TAG_ID="${TARGET_TAG_ID:-0}"
USE_FIRST_DETECTION="${USE_FIRST_DETECTION:-false}"
THETA_OFFSET_RAD="${THETA_OFFSET_RAD:-0.0}"
ENABLE_THETA_OFFSET="${ENABLE_THETA_OFFSET:-true}"
THETA_DEADBAND_RAD="${THETA_DEADBAND_RAD:-0.0}"
ENABLE_THETA_DEADBAND="${ENABLE_THETA_DEADBAND:-true}"
DISTANCE_OFFSET="${DISTANCE_OFFSET:-0.6}"
MIN_DISTANCE="${MIN_DISTANCE:-0.0}"
USE_ODOM_FALLBACK="${USE_ODOM_FALLBACK:-false}"
ODOM_TOPIC="${ODOM_TOPIC:-/odom}"

echo "Go2 D435 image topic:       $GO2_IMAGE_TOPIC"
echo "Go2 D435 camera_info topic: $GO2_CAMERA_INFO_TOPIC"
echo "AprilTag detections topic:  $TAG_DETECTIONS_TOPIC"
echo "Polar output topic:         $POLAR_TOPIC"
echo "Use odom fallback:          $USE_ODOM_FALLBACK"

roslaunch go_nav go2_detection_full.launch \
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
  min_distance:="$MIN_DISTANCE" \
  use_odom_fallback:="$USE_ODOM_FALLBACK" \
  odom_topic:="$ODOM_TOPIC"
