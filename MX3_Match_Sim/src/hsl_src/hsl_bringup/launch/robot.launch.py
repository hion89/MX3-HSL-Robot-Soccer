#!/usr/bin/env python3
"""실기 1 대 = mx3_src 하위 계층 (bringup: 다이나믹셀·IMU·보행/킥 러너·목) + 카메라/YOLO + GC + hsl 상위 계층.
★ 아직 실기에서 돌려 보지 않았다 (시뮬에서만 검증). Humble 에서 빌드만 확인할 것 ★

  ros2 launch hsl_bringup robot.launch.py              # robot = hostname 앞부분 → robot_params.yaml
  ros2 launch hsl_bringup robot.launch.py player_number:=3

match.launch.py 에서 빠지는 것: ekf_localization, pose_goal, depth_obstacle, neck_ball_tracker,
  ball_follow, vision_servo, strategy, team_comm  (hsl 의 world_model·behavior·team_comm 이 대체)
"""
import os
import socket

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _inc(pkg, f, args=None):
    path = os.path.join(get_package_share_directory(pkg), "launch", f)
    return IncludeLaunchDescription(PythonLaunchDescriptionSource(path), launch_arguments=(args or {}).items())


def _setup(context):
    robot = LaunchConfiguration("robot").perform(context).strip() or \
        socket.gethostname().split("-")[0].split(".")[0]
    cfg = os.path.join(get_package_share_directory("humanoid_bringup"), "config", "robot_params.yaml")
    with open(cfg, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    merged = dict(data.get("common", {}) or {})
    merged.update(data.get(robot, {}) or {})
    team = LaunchConfiguration("team_number").perform(context).strip() or str(merged.get("team_number", 35))
    player = LaunchConfiguration("player_number").perform(context).strip() or str(merged.get("player_number", 1))
    return [
        _inc("humanoid_bringup", "bringup.launch.py", {"use_kick": "true"}),
        _inc("humanoid_perception", "perception_bringup.launch.py"),
        _inc("humanoid_game_controller", "game_controller.launch.py",
             {"robot": robot, "team_number": team, "player_number": player}),
        _inc("hsl_bringup", "hsl_stack.launch.py", {
            "team_number": team, "player_number": player, "field_profile": LaunchConfiguration("field_profile")}),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("robot", default_value=""),
        DeclareLaunchArgument("team_number", default_value=""),
        DeclareLaunchArgument("player_number", default_value=""),
        DeclareLaunchArgument("field_profile", default_value="s_field"),
        OpaqueFunction(function=_setup),
    ])
