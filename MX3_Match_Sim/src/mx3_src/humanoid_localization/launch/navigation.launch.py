#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  navigation.launch.py
#  Nav2 스택 + nav_cmd_bridge 를 띄운다. (localization 은 나중 → map→odom static)
#
#  띄우는 것:
#    - map_server (빈 필드 맵)
#    - planner_server (NavFn)
#    - controller_server (RPP) — 출력 /cmd_vel 을 /cmd_vel_nav_raw 로 remap
#    - behavior_server (spin/wait)
#    - bt_navigator
#    - lifecycle_manager (위 노드들 자동 활성화)
#    - nav_cmd_bridge — /cmd_vel_nav_raw(m/s) → /cmd_vel_nav(명령값)
#
#  전제(별도 실행):
#    - 하드웨어 + 보행 파이프라인:  control.launch.py (twist_mux/actor 등)
#    - 오도메트리 + map→odom + 필드: odom_display.launch.py (또는 walk_odometry + static tf)
#
#  사용:
#    ros2 launch humanoid_localization navigation.launch.py
#    그 뒤 RViz 의 "Nav2 Goal" 로 목적지를 찍으면 그쪽으로 보행한다.
# =============================================================================

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('humanoid_localization')
    params = os.path.join(share, 'config', 'nav2_params.yaml')
    map_yaml = os.path.join(share, 'config', 'map', 'empty_field.yaml')

    # lifecycle_manager 가 관리할 노드 목록
    lifecycle_nodes = [
        'map_server', 'planner_server', 'controller_server',
        'behavior_server', 'bt_navigator',
    ]

    map_server = Node(
        package='nav2_map_server', executable='map_server', name='map_server',
        output='screen',
        parameters=[params, {'yaml_filename': map_yaml}],  # 절대경로로 덮어씀
    )
    planner = Node(
        package='nav2_planner', executable='planner_server', name='planner_server',
        output='screen', parameters=[params],
    )
    controller = Node(
        package='nav2_controller', executable='controller_server',
        name='controller_server', output='screen', parameters=[params],
        # ★ RPP 가 내는 cmd_vel(m/s)을 브릿지 입력 토픽으로 돌린다
        remappings=[('cmd_vel', '/cmd_vel_nav_raw')],
    )
    behavior = Node(
        package='nav2_behaviors', executable='behavior_server',
        name='behavior_server', output='screen', parameters=[params],
        remappings=[('cmd_vel', '/cmd_vel_nav_raw')],  # 회복 동작도 동일 경로
    )
    bt_nav = Node(
        package='nav2_bt_navigator', executable='bt_navigator',
        name='bt_navigator', output='screen', parameters=[params],
    )
    lifecycle = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_navigation', output='screen',
        parameters=[{'autostart': True}, {'node_names': lifecycle_nodes}],
    )

    # m/s → 정책 명령값 브릿지 (출력 /cmd_vel_nav 가 twist_mux navigation 입력)
    bridge = Node(
        package='humanoid_localization', executable='nav_cmd_bridge_node',
        name='nav_cmd_bridge_node', output='screen',
        parameters=[os.path.join(share, 'config', 'odom_calib.yaml')],
    )

    return LaunchDescription([
        map_server, planner, controller, behavior, bt_nav, lifecycle, bridge,
    ])
