# -*- coding: utf-8 -*-
# =============================================================================
#  ball_follow.launch.py
#
#  ball_follow_nav_node(Phase 2: /pose_goal 발행으로 공 근처까지 접근)를 단독 실행하는 런치.
#  파라미터는 config/ball_follow.yaml 에서 읽는다(튜닝은 그 파일 수정 후 노드 재시작).
#  필드에서 자주 바꾸는 정렬 모드·공격 골대(사이드)만 런치 인자로 노출해 YAML 값을 덮어쓴다.
#
#  [전제]
#   - ball_map_projector_node 가 /ball/map_point 를 발행 중이어야 함.
#   - pose_goal_controller_node 가 떠 있어야 함(ball_follow 가 enable 을 소유·토글).
#   - 로컬라이제이션 스택이 map→base_link TF 를 발행 중이어야 함.
#   ※Nav2 불필요 — pose_goal_controller 가 cmd_vel_nav 슬롯을 구동한다.
#
#  [실행]
#     ros2 launch humanoid_localization ball_follow.launch.py
#   좌표계는 팀 상대(상대 골=+x 고정)라 하프타임에 부호를 뒤집지 않는다.
#   다른 필드 프로파일(연습장)일 때만 값 override:
#     ros2 launch humanoid_localization ball_follow.launch.py target_goal_x:=1.95
#   레거시(골대 무시, 로봇→공 직선상 접근)로 돌리려면:
#     ros2 launch humanoid_localization ball_follow.launch.py align_to_goal:=false
#
#  [튜닝]
#   standoff/도착거리/타이밍 등은 config/ball_follow.yaml 수정 후 재시작, 또는 라이브:
#     ros2 param set /ball_follow_nav_node arrived_distance_m 0.45
#     ros2 param set /ball_follow_nav_node standoff_m 0.40
# =============================================================================

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ── config/ball_follow.yaml 경로(설치된 share 디렉토리에서 조회) ──
    cfg = os.path.join(
        get_package_share_directory('humanoid_localization'),
        'config', 'ball_follow.yaml')

    # 필드에서 자주 바꾸는 항목만 런치 인자로 노출(YAML 값을 override).
    align_to_goal = LaunchConfiguration('align_to_goal')
    target_goal_x = LaunchConfiguration('target_goal_x')
    target_goal_y = LaunchConfiguration('target_goal_y')

    return LaunchDescription([
        # ── 정렬 모드 / 공격 골대(사이드) ──────────────────────────
        # default_value 는 YAML 과 동일하게 둔다(인자 미지정 시 YAML 값과 일치).
        DeclareLaunchArgument(
            'align_to_goal', default_value='true',
            description='true=골대–공–로봇 일직선 정렬(공 뒤에서 골대를 봄), '
                        'false=레거시(로봇→공 직선상 접근)'),
        DeclareLaunchArgument(
            'target_goal_x', default_value='4.5',
            description='공격 골대 중심 x[m] (map). 상대 골=+x 고정(팀 상대). S-Field=4.5, 연습장=1.95'),
        DeclareLaunchArgument(
            'target_goal_y', default_value='0.0',
            description='공격 골대 중심 y[m] (map). 골 중심은 보통 0.0'),

        Node(
            package='humanoid_localization',
            executable='ball_follow_nav_node',
            name='ball_follow_nav_node',
            output='screen',
            # YAML 먼저 로드 → 런치 인자로 일부만 덮어씀(인자 우선).
            # standoff_m/arrived_distance_m/타이밍/토픽 등 나머지는 전부 YAML 값 사용.
            parameters=[
                cfg,
                {
                    'align_to_goal': align_to_goal,
                    'target_goal_x': target_goal_x,
                    'target_goal_y': target_goal_y,
                },
            ],
        ),
    ])
