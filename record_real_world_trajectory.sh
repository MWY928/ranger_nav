#!/usr/bin/env bash

# =========================================================
# record_real_world_trajectory.sh
#
# Purpose:
#   Record real-world deployment trajectories from the FALCON ROS bridge.
#
# This script records:
#   1. Goal polar coordinates: distance r in meters and angle theta in radians
#   2. Depth observations in meters
#   3. Policy actions and action probabilities
#
# Notes:
#   - Replay dumping is enabled for later replay/ sim-real comparison.
#   - This script is for real-robot deployment data recording.
# =========================================================

set -e

CHECKPOINT="weights/ours_hm3d_val_best.pth"
REPLAY_DUMP_LIMIT=60

rosrun ranger_nav falcon_ros_bridge.py \
  --checkpoint "$CHECKPOINT" \
  --replay_dump_enabled \
  --replay_dump_limit "$REPLAY_DUMP_LIMIT"
