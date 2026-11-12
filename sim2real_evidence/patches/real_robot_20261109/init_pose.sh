#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
source /home/robo04/sim2real_ws/install/setup.bash
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=1
# Practice drawing: our goal line, exact midpoint of left T marks y=0.65 and 1.085.
# Place the robot at x=-1.95, y=+0.8675, facing the opponent goal (yaw=0).
# Run after the physical START button, which resets localization.
timeout 15s ros2 topic pub --once \
  /initialpose geometry_msgs/msg/PoseWithCovarianceStamped \
  "{header: {frame_id: map}, pose: {pose: {position: {x: -1.95, y: 0.8675, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}}"
