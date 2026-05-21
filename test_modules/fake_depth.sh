#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /opt/ros/noetic/setup.bash
if [ -f /home/mobile/catkin_ws/devel/setup.bash ]; then
  source /home/mobile/catkin_ws/devel/setup.bash
fi
if [ -f /home/mobile/ranger_ws/devel/setup.bash ]; then
  source /home/mobile/ranger_ws/devel/setup.bash
fi

python "$SCRIPT_DIR/fake_depth_pub.py" \
  --topic /camera/aligned_depth_to_color/image_raw \
  --encoding 16UC1 \
  --mode fixed \
  --depth_m 0.2 \
  --rate 20
