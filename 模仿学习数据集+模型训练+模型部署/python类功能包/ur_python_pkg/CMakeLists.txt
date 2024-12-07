cmake_minimum_required(VERSION 3.0.2)
project(ur_python_pkg)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  actionlib
  control_msgs
  diagnostic_msgs
  gazebo_msgs
  geometry_msgs
  message_generation
  moveit_msgs
  robotiq_ft_sensor
  roscpp
  rospy
  sensor_msgs
  shape_msgs
  std_msgs
  std_srvs
  tf
  tf2_geometry_msgs
  trajectory_msgs
)

find_package(Eigen3 REQUIRED CONFIG)

catkin_python_setup()

add_message_files(
  FILES
  OmniButtonEvent.msg
  Robotiq2FGripper_robot_input.msg
  Robotiq2FGripper_robot_output.msg
)
generate_messages(
  DEPENDENCIES
  std_msgs
  # OmniButtonEvent.msg
  # Robotiq2FGripper_robot_input.msg
  # Robotiq2FGripper_robot_output.msg
)
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES ur_planning
   CATKIN_DEPENDS roscpp rospy std_msgs message_runtime gazebo_msgs
#  DEPENDS system_lib
)
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)
