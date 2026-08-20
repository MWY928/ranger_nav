#!/usr/bin/env bash

# =========================================================
# record_real_world_trajectory.sh
#
# Purpose:
#   Record real-world navigation algo deployment trajectories from the FALCON ROS bridge.
#
# This script records:
#   1. Goal polar coordinates: distance r in meters and angle theta in radians
#   2. Depth observations in meters
#   3. Policy actions and action probabilities
#
# Notes:
#   - Replay dumping can be used for later replay/ sim-real comparison.
#   - Or used as a debug tool
#   - This script is for real-robot deployment data recording.
# =========================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
RESULTS_DIR="$REPO_ROOT/test_modules/test_results/004"
CHECKPOINT="$REPO_ROOT/weights/falcon_bc_70traj_action_head_lstm004.pth"
REPLAY_DUMP_LIMIT=900
REPLAY_DUMP_ROOT="$RESULTS_DIR/traj/bridge_policy_replay/00406"
RUN_ID="${1:-$(date +%Y%m%d_%H%M%S)}"
REPLAY_DUMP_DIR="${REPLAY_DUMP_ROOT}/${RUN_ID}"
DEBUG_DEPTH_DUMP_DIR="$RESULTS_DIR/bridge_depth_samples"

mkdir -p "$REPLAY_DUMP_DIR"
mkdir -p "$DEBUG_DEPTH_DUMP_DIR"
echo "Recording replay samples to: $REPLAY_DUMP_DIR"
echo "Recording depth debug samples to: $DEBUG_DEPTH_DUMP_DIR"
echo "Replay dump limit: $REPLAY_DUMP_LIMIT"

ACTION_TOPIC="${ACTION_TOPIC:-/falcon/action_id}"
DEPTH_TOPIC="${DEPTH_TOPIC:-/camera/depth/image_rect_raw}"

exec python "$REPO_ROOT/sensor/falcon_ros_bridge.py" \
  --checkpoint "$CHECKPOINT" \
  --depth_topic "$DEPTH_TOPIC" \
  --polar_topic /tag_polar \
  --action_topic "$ACTION_TOPIC" \
  --debug_mapping \
  --debug_depth \
  --debug_depth_dump_dir "$DEBUG_DEPTH_DUMP_DIR" \
  --replay_dump_enabled \
  --replay_dump_dir "$REPLAY_DUMP_DIR" \
  --replay_dump_limit "$REPLAY_DUMP_LIMIT"
