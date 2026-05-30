cmake_minimum_required(VERSION 3.0.2)
project(traversability_planner)

find_package(catkin REQUIRED COMPONENTS
  rospy
  nav_msgs
  geometry_msgs
  sensor_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/terrain_aware_planner_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

