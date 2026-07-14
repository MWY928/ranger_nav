#!/bin/bash
set -e

source /opt/ros/noetic/setup.bash
source /home/mobile/catkin_ws/devel/setup.bash
source /home/mobile/ranger_ws/devel/setup.bash

source /home/mobile/miniconda3/etc/profile.d/conda.sh
conda activate simple_nav

roslaunch ranger_nav tag_follower.launch
