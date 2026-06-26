cmake_minimum_required(VERSION 3.0.2)
project(go_nav)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/unitree_action_mapper.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
