#!/usr/bin/env python3
"""hsl 상위 계층 (실기·시뮬 공통): 다리 오도메트리 → 자기위치·공 추적 → 팀 통신 → 행동.

  team_number / player_number / field_profile 을 받는다. 진입 위치(사이드라인) x_inset = 0.7 × 선수 번호
  (기존 reentry_positions.yaml 의 robo01~04 값과 같은 규칙).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _nodes(context):
    team = int(LaunchConfiguration("team_number").perform(context))
    player = int(LaunchConfiguration("player_number").perform(context))
    field = LaunchConfiguration("field_profile").perform(context)
    out = LaunchConfiguration("node_output").perform(context)
    cfg = os.path.join(get_package_share_directory("hsl_bringup"), "config", "hsl.yaml")
    urdf = os.path.join(get_package_share_directory("hsl_world_model"), "urdf", "XM_robot04.urdf")
    inset = float(LaunchConfiguration("walk_in_x_inset").perform(context) or 0.7 * player)
    common = {"field_profile": field, "player_number": player}
    rate = LaunchConfiguration("team_send_rate_hz").perform(context)
    comm = {"send_rate_hz": float(rate)} if rate else {}       # 비우면 hsl.yaml (2 Hz, 메시지 예산)
    event = LaunchConfiguration("team_event_send").perform(context)
    if event:
        comm["event_send"] = event.lower() == "true"
    return [
        Node(package="hsl_world_model", executable="leg_odometry", name="hsl_leg_odometry", output=out,
             parameters=[cfg, {"urdf_path": urdf}]),
        Node(package="hsl_world_model", executable="self_localizer", name="hsl_self_localizer", output=out,
             parameters=[cfg, common, {"walk_in_x_inset": inset}]),
        Node(package="hsl_world_model", executable="ball_tracker", name="hsl_ball_tracker", output=out,
             parameters=[cfg]),
        Node(package="hsl_team_comm", executable="team_comm", name="hsl_team_comm", output=out,
             parameters=[cfg, {"team_number": team, "player_number": player}, comm]),
        Node(package="hsl_behavior", executable="behavior", name="hsl_behavior", output=out,
             parameters=[cfg, common, {"goalkeeper_number": int(LaunchConfiguration("goalkeeper_number")
                                                                .perform(context))}]),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("team_number", default_value="35"),
        DeclareLaunchArgument("player_number", default_value="1"),
        DeclareLaunchArgument("goalkeeper_number", default_value="1"),
        DeclareLaunchArgument("field_profile", default_value="s_field"),
        DeclareLaunchArgument("walk_in_x_inset", default_value="", description="비우면 0.7 × 선수 번호"),
        DeclareLaunchArgument("node_output", default_value="screen"),
        DeclareLaunchArgument("team_send_rate_hz", default_value="", description="비우면 hsl.yaml (시험용 덮어쓰기)"),
        DeclareLaunchArgument("team_event_send", default_value="", description="true/false, 비우면 hsl.yaml"),
        OpaqueFunction(function=_nodes),
    ])
