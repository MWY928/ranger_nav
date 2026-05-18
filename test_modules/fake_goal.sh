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

python "$SCRIPT_DIR/fake_polar_goal_pub.py" \
  --topic /tag_polar \
  --mode fixed \
  --r 2.0 \
  --theta 0.8 \
  --rate 20
