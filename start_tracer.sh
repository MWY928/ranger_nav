#!/bin/bash
set -e

pkill -f tracer_base_node
sudo ip link set can0 down
sudo modprobe -r gs_usb

source /opt/ros/noetic/setup.bash
source /home/mobile/catkin_ws/devel/setup.bash

sudo modprobe gs_usb
rosrun tracer_bringup bringup_can2usb.bash

sleep 1

roslaunch tracer_bringup tracer_robot_base.launch
