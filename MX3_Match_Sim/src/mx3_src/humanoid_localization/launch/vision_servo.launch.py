# -*- coding: utf-8 -*-
# =============================================================================
#  vision_servo.launch.py
#
#  visual_servo_kick_node(Phase 3: 비전 서보 정렬 + 킥)를 단독 실행하는 런치.
#  파라미터는 config/vision_servo.yaml 에서 읽는다(튜닝은 그 파일을 수정).
#  자주 바꾸는 런타임 토글만 런치 인자로 노출해 YAML 값을 덮어쓴다.
#
#  [전제]
#   - yolo_detector_3d_node 가 /perception/detections_3d 발행 중(공/골포스트).
#   - RL 보행정책 actor_onnx_walk_residual_kick 가 /cmd_vel 추종 중.
#   - twist_mux: cmd_vel_behavior → /cmd_vel 경로 활성.
#   - TF: camera_color_optical_frame→base_link(필수), localization 폴백 시 map→base_link.
#
#  [Phase 2 → Phase 3 핸드오프]
#   비전 서보=cmd_vel_behavior(priority 10) < Nav2=cmd_vel_nav(priority 20).
#   통합 실행 시 Phase 2 nav goal 을 취소한 뒤 비전 서보를 enable 해야 cmd 가 먹는다.
#
#  [실행]
#     ros2 launch humanoid_localization vision_servo.launch.py
#   단독 테스트(켜자마자):
#     ros2 launch humanoid_localization vision_servo.launch.py auto_start:=true
#   localization 폴백 골 방향은 팀 상대(상대 골=+x 고정)라 뒤집지 않는다:
#     ros2 launch humanoid_localization vision_servo.launch.py \
#         goal_source:=localization auto_start:=true
#
#  [튜닝]
#   속도/게인/허용오차 등은 config/vision_servo.yaml 수정 후 노드 재시작.
#   라이브로도 가능:  ros2 param set /visual_servo_kick_node vx_max 0.30
# =============================================================================

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # config/vision_servo.yaml 경로(설치본 share 디렉터리).
    cfg = os.path.join(
        get_package_share_directory('humanoid_localization'),
        'config', 'vision_servo.yaml')

    goal_source = LaunchConfiguration('goal_source')
    target_goal_x = LaunchConfiguration('target_goal_x')
    target_goal_y = LaunchConfiguration('target_goal_y')
    auto_start = LaunchConfiguration('auto_start')
    continuous_mode = LaunchConfiguration('continuous_mode')

    return LaunchDescription([
        # ── 런타임 토글(YAML 값을 덮어씀) ──────────────────────────
        DeclareLaunchArgument('goal_source', default_value='auto',
                              description="골대 방위 소스: 'auto'/'vision'/'localization'"),
        DeclareLaunchArgument('target_goal_x', default_value='4.5',
                              description='localization 폴백 상대(공격)골대 x[m]. '
                                          'S-Field 골라인=4.5(연습장 1.95). '
                                          '상대 골=+x 고정(팀 상대) — 사이드 배정에 따라 뒤집지 말 것'),
        DeclareLaunchArgument('target_goal_y', default_value='0.0',
                              description='localization 폴백 공격골대 y[m]'),
        DeclareLaunchArgument('auto_start', default_value='false',
                              description='true=시작 즉시 동작(단독). false=enable 대기(통합)'),
        DeclareLaunchArgument('continuous_mode', default_value='false',
                              description='true=킥 후 자동 재탐색 반복'),

        Node(
            package='humanoid_localization',
            executable='visual_servo_kick_node',
            name='visual_servo_kick_node',
            output='screen',
            # YAML 먼저 로드 → 그 뒤 런치 인자로 일부만 덮어씀(인자 우선).
            parameters=[
                cfg,
                {
                    'goal_source': goal_source,
                    'target_goal_x': target_goal_x,
                    'target_goal_y': target_goal_y,
                    'auto_start': auto_start,
                    'continuous_mode': continuous_mode,
                },
            ],
        ),
    ])
