cmake_minimum_required(VERSION 3.0.2)
project(ranger_nav)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  src/realsense_stream.py
  src/polar_distance.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
