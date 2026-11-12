#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
neck_ball_tracker.launch.py
===========================

목으로 공을 추적하는 neck_ball_tracker_node 만 띄우는 launch.

[ 전제 — 이 launch 전에 떠 있어야 함 ]
  · neck_control_node (목 명령/현재각 — 2026-08 목 이관 후 한 노드로 통합)
  · 목 토크 ON (/neck/torque_enable True)
  · yolo_detector_3d_node (/perception/detections_3d, 광학프레임)
  보통 bringup 에서 위가 이미 떠 있으므로, 추적만 추가로 올릴 때 이 launch 를 쓴다.

사용:
  # 기본(패키지 config 의 yaml 사용)
  ros2 launch humanoid_serial_bridge neck_ball_tracker.launch.py

  # 다른 파라미터 파일
  ros2 launch humanoid_serial_bridge neck_ball_tracker.launch.py \
      params:=/절대/경로/neck_ball_tracker.yaml

확인:
  ros2 topic echo /neck/tracking_state         # TRACKING / RETURN / NEUTRAL
  ros2 topic echo /neck/track_error_deg        # [수평, 수직] 각오차(deg)
  ros2 topic echo /neck/yaw_deg                 # 나가는 명령
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # 패키지 share 안의 기본 파라미터 파일 경로.
    # setup.py 의 data_files 가 config/*.yaml 을 share/<pkg>/config/ 로 install 함.
    default_params = os.path.join(
        get_package_share_directory('humanoid_serial_bridge'),
        'config', 'neck_ball_tracker.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params', default_value=default_params,
            description='neck_ball_tracker_node 파라미터 yaml 경로'),

        Node(
            package='humanoid_serial_bridge',
            executable='neck_ball_tracker_node',
            name='neck_ball_tracker_node',
            output='screen',
            parameters=[LaunchConfiguration('params')],
        ),
    ])
