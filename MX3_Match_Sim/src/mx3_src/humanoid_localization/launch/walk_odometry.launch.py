#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  walk_odometry.launch.py
#  walk_odometry_node 를 config/odom_calib.yaml 파라미터와 함께 실행한다.
#
#  사용:
#     ros2 launch humanoid_localization walk_odometry.launch.py
#
#  주의: 이 노드는 /imu/data 와 /cmd_vel 을 구독하므로, 별도로
#        dynamixel_system.launch.py(IMU 발행)와 보행/명령 노드가 떠 있어야
#        실제 의미 있는 오도메트리가 나온다.
# =============================================================================

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('humanoid_localization')

    # install 디렉토리에 복사된 calib yaml 의 절대경로
    # (setup.py 의 data_files 에 config 를 등록했기 때문에 여기서 찾을 수 있다)
    calib_yaml = os.path.join(pkg_share, 'config', 'odom_calib.yaml')

    walk_odometry = Node(
        package='humanoid_localization',
        executable='walk_odometry_node',
        name='walk_odometry_node',   # ★ yaml 최상위 키와 일치해야 파라미터가 적용됨
        output='screen',
        parameters=[calib_yaml],     # yaml 파일을 통째로 파라미터로 주입
        # 토픽 이름을 바꿔야 할 경우 remappings 로 처리:
        # remappings=[('/cmd_vel', '/policy/cmd_vel')],
    )

    return LaunchDescription([walk_odometry])
