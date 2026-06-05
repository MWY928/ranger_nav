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

exec python "$REPO_ROOT/sensor/falcon_ros_bridge.py" \
  --checkpoint "$REPO_ROOT/weights/falcon_bc_70traj_action_head_lstm2.pth" \
  --depth_topic /camera/aligned_depth_to_color/image_raw \
  --polar_topic /tag_polar \
  --cmd_vel_topic /cmd_vel \
  --debug_mapping \
  --debug_depth \
  --debug_depth_dump_dir "$RESULTS_DIR/bridge_depth_samples"

  
