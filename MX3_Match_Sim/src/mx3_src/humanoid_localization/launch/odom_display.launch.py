#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  odom_display.launch.py
#  RViz 로 "5x3m 필드 위에서 움직이는 로봇"을 보기 위한 시각화 통합 launch.
#
#  띄우는 것:
#    1) robot_tf_bringup (humanoid_perception) → robot_state_publisher
#         base_link 아래 전 관절 TF + /robot_description
#    2) map→odom static TF (use_static_map_tf=true 일 때)
#         아직 마커 localization 이 없으므로 임시 identity. localization 을 붙이면
#         그 노드가 map→odom 을 발행하므로 use_static_map_tf:=false 로 끈다.
#    3) field_lines_node (humanoid_localization) → /field_markers (필드 라인/골대)
#    4) walk_odometry (humanoid_localization) → odom→base_link TF + /odom
#    5) rviz2 → walk_odometry.rviz (Fixed Frame=map)
#
#  전제(별도 터미널):
#    - ros2 launch humanoid_serial_bridge dynamixel_system.launch.py  (IMU + joint_states)
#    - actor_onnx_runner + keyboard_teleop_node                      (실제 보행 + 명령)
#
#  사용:
#    ros2 launch humanoid_localization odom_display.launch.py
#    # 로봇 시작 지점이 필드 중앙이 아니라면(예: 자기 진영 골 앞):
#    ros2 launch humanoid_localization odom_display.launch.py map_x:=-2.0 map_yaw:=0.0
#    # 책상에서 모델만 점검:
#    ros2 launch humanoid_localization odom_display.launch.py use_jsp_gui:=true
# =============================================================================

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    loc_share = get_package_share_directory('humanoid_localization')
    perc_share = get_package_share_directory('humanoid_perception')

    rviz_config = os.path.join(loc_share, 'rviz', 'walk_odometry.rviz')
    field_yaml = os.path.join(loc_share, 'config', 'field.yaml')

    # ---- launch 인자 ----
    use_jsp_gui = LaunchConfiguration('use_jsp_gui')
    enable_camera = LaunchConfiguration('enable_camera')
    use_static_map_tf = LaunchConfiguration('use_static_map_tf')
    map_x = LaunchConfiguration('map_x')      # map→odom offset (로봇 시작 위치 보정)
    map_y = LaunchConfiguration('map_y')
    map_yaw = LaunchConfiguration('map_yaw')

    # 1) 로봇 TF
    robot_tf = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(perc_share, 'launch', 'robot_tf_bringup.launch.py')),
        launch_arguments={
            'use_jsp_gui': use_jsp_gui,
            'enable_camera': enable_camera,
        }.items(),
    )

    # 2) map → odom static TF (localization 전 임시)
    #    로봇을 필드 중앙이 아닌 곳에서 시작하려면 map_x/map_y/map_yaw 로 odom 원점을
    #    필드 좌표상의 시작 위치에 놓는다. (예: 왼쪽 골 앞이면 map_x:=-2.0)
    static_map = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_map_to_odom',
        condition=IfCondition(use_static_map_tf),
        arguments=[
            '--x', map_x, '--y', map_y, '--z', '0',
            '--yaw', map_yaw, '--pitch', '0', '--roll', '0',
            '--frame-id', 'map', '--child-frame-id', 'odom',
        ],
    )

    # 3) 필드 라인
    #    field_profile 을 주입해야 9x6 대회 규격으로도 라인이 그려진다.
    #    (미주입 시 field.yaml 기본값 practice=3.9x2.6 로 떠 버린다.)
    field = Node(
        package='humanoid_localization',
        executable='field_lines_node',
        name='field_lines_node',
        output='screen',
        parameters=[
            field_yaml,
            {'field_profile': LaunchConfiguration('field_profile')},
        ],
    )

    # 4) 보행 오도메트리
    walk_odom = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(loc_share, 'launch', 'walk_odometry.launch.py')),
    )

    # 5) RViz2
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_jsp_gui', default_value='false',
            description='true 면 관절을 슬라이더로 조작(로봇 없이 점검용)'),
        DeclareLaunchArgument('enable_camera', default_value='false',
            description='RealSense 실행 여부(시각화 데모는 기본 false)'),
        DeclareLaunchArgument('use_static_map_tf', default_value='true',
            description='map→odom 임시 static TF 발행. localization 붙으면 false 로 끈다'),
        DeclareLaunchArgument('map_x', default_value='0.0',
            description='map→odom x offset(=odom 원점의 필드 내 x 위치)'),
        DeclareLaunchArgument('map_y', default_value='0.0',
            description='map→odom y offset'),
        DeclareLaunchArgument('map_yaw', default_value='0.0',
            description='map→odom yaw offset(rad)'),
        DeclareLaunchArgument('field_profile', default_value='practice',
            description='경기장 규격: practice(연습장 3.9x2.6) / s_field(대회 9x6)'),
        robot_tf,
        static_map,
        field,
        walk_odom,
        rviz,
    ])