#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source /opt/ros/noetic/setup.bash
if [ -f /home/mobile/catkin_ws/devel/setup.bash ]; then
  source /home/mobile/catkin_ws/devel/setup.bash
fi
if [ -f /home/mobile/ranger_ws/devel/setup.bash ]; then
  source /home/mobile/ranger_ws/devel/setup.bash
fi

source /home/mobile/miniconda3/etc/profile.d/conda.sh
conda activate falcon

RESULTS_DIR="$REPO_ROOT/test_modules/test_results"
mkdir -p "$RESULTS_DIR"

# This test bridge script is designed to debuging the navigation algo, which:
# - --debug_mapping prints each inference step to ROS logs, including goal
#   polar input, selected action id, action topic, and action probabilities.
# - --debug_depth prints depth preprocessing stats to ROS logs, including raw
#   encoding/shape/dtype, valid/min/max/mean/p50/p95/zero ratios, crop shape,
#   and normalized Falcon input shape/range.
# - --debug_depth_dump_dir saves one depth sample only, under:
#     test_modules/test_results/bridge_depth_samples/
#   The dump includes raw/meter/normalized depth as .npy and .csv files,
#   preview .png images for each stage, and a metadata .json with the same
#   depth statistics. It saves only the first frame to avoid large log files.
# - This script does not enable --replay_dump_enabled, so it does not save
#   policy replay .npz/.json files.
ACTION_TOPIC="${ACTION_TOPIC:-/falcon/action_id}"

exec python "$REPO_ROOT/sensor/falcon_ros_bridge.py" \
  --checkpoint "$REPO_ROOT/weights/falcon_bc_70traj_action_head_lstm2.pth" \
  --depth_topic /camera/aligned_depth_to_color/image_raw \
  --polar_topic /tag_polar \
  --action_topic "$ACTION_TOPIC" \
  --debug_mapping \
  --debug_depth \
  --debug_depth_dump_dir "$RESULTS_DIR/bridge_depth_samples"

  
