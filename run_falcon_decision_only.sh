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

CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/weights/falcon_bc_70traj_action_head_lstm0045.pth}"
DEPTH_TOPIC="${DEPTH_TOPIC:-/camera/aligned_depth_to_color/image_raw}"
RELATIVE_GOAL_TOPIC="${RELATIVE_GOAL_TOPIC:-/nav_bridge/relative_goal}"
GOAL_VALID_TOPIC="${GOAL_VALID_TOPIC:-/nav_bridge/goal_valid}"
DECISION_TOPIC="${DECISION_TOPIC:-/falcon/decision_cmd}"

exec python "$REPO_ROOT/sensor/falcon_nav_bridge.py" \
  --checkpoint "$CHECKPOINT" \
  --depth_topic "$DEPTH_TOPIC" \
  --relative_goal_topic "$RELATIVE_GOAL_TOPIC" \
  --goal_valid_topic "$GOAL_VALID_TOPIC" \
  --command_topic "$DECISION_TOPIC" \
  --debug_mapping \
  --debug_depth "$@"
