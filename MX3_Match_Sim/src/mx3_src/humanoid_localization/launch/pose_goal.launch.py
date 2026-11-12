# -*- coding: utf-8 -*-
# =============================================================================
#  pose_goal.launch.py
#
#  pose_goal_controller_node(제네릭 go-to-pose + 회피)를 단독 실행하는 런치.
#  파라미터는 config/pose_goal_controller.yaml 에서 읽는다(튜닝은 그 파일 수정).
#  자주 바꾸는 토글만 런치 인자로 노출해 YAML 값을 덮어쓴다.
#
#  [전제]
#   - RL 보행정책 actor 가 /cmd_vel 추종 중.
#   - twist_mux: cmd_vel_nav(navigation, priority 20) → /cmd_vel 경로 활성.
#   - TF: (목표 frame_id) → base_link. odom→base_link(walk_odometry) 또는
#     map→base_link(localization)가 살아 있어야 한다.
#   - (선택) depth_obstacle_detector_node 가 /obstacle/avoid_hint 발행 — 없으면
#     회피 없이 순수 추종으로 동작(graceful degrade).
#
#  [목표 주기]
#   누군가 goal_topic 으로 PoseStamped 를 발행해야 움직인다(set-point).
#   예: ros2 topic pub --once /pose_goal geometry_msgs/PoseStamped \
#         "{header: {frame_id: 'odom'}, pose: {position: {x: 1.0, y: 0.0},
#           orientation: {w: 1.0}}}"
#
#  [실행]
#     ros2 launch humanoid_localization pose_goal.launch.py
#   회피 끄고 추종만 보기:
#     ros2 launch humanoid_localization pose_goal.launch.py avoid_enable:=false
#   켜자마자 enable(goal 오기 전엔 IDLE):
#     ros2 launch humanoid_localization pose_goal.launch.py auto_start:=true
#
#  [튜닝]
#   게인/속도/임계/회피는 config/pose_goal_controller.yaml 수정 후 재시작, 또는
#   라이브:  ros2 param set /pose_goal_controller_node react_dist 1.2
# =============================================================================

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    cfg = os.path.join(
        get_package_share_directory('humanoid_localization'),
        'config', 'pose_goal_controller.yaml')

    auto_start = LaunchConfiguration('auto_start')
    avoid_enable = LaunchConfiguration('avoid_enable')
    cmd_vel_topic = LaunchConfiguration('cmd_vel_topic')

    return LaunchDescription([
        DeclareLaunchArgument('auto_start', default_value='false',
                              description='true=시작 즉시 enable(goal 대기). false=enable 대기'),
        DeclareLaunchArgument('avoid_enable', default_value='true',
                              description='true=장애물 회피 합성 on'),
        DeclareLaunchArgument('cmd_vel_topic', default_value='cmd_vel_nav',
                              description='출력 토픽(twist_mux 슬롯). 기본 navigation'),

        Node(
            package='humanoid_localization',
            executable='pose_goal_controller_node',
            name='pose_goal_controller_node',
            output='screen',
            # YAML 먼저 로드 → 런치 인자로 일부만 덮어씀(인자 우선).
            parameters=[
                cfg,
                {
                    'auto_start': auto_start,
                    'avoid_enable': avoid_enable,
                    'cmd_vel_topic': cmd_vel_topic,
                },
            ],
        ),
    ])
