# -*- coding: utf-8 -*-
# =============================================================================
#  depth_obstacle.launch.py
#
#  단거리 깊이 장애물 검출(depth_obstacle_detector_node)을 띄운다.
#  거리구간별 바닥추정 + density 필터로 전방 코리도에 솟은 물체를 잡고,
#  /obstacle/avoid_hint (Vector3Stamped) 로 회피 힌트를 발행 → pose_goal_controller 소비.
#
#  [전제]
#   - 깊이 카메라 구동: /camera/camera/aligned_depth_to_color/image_raw + color/camera_info
#   - TF: camera_color_optical_frame → base_link (robot_state_publisher; 목 틸트 포함).
#     ★카메라 높이는 TF가 처리 — 바닥추정이 데이터 기반이라 카메라 높이 파라미터 불필요.
#   - 회피가 의미 있으려면 네비 구간 동안 목이 전방을 봐야 함.
#
#  [실행]
#     ros2 launch humanoid_perception depth_obstacle.launch.py
#   거리 바꾸기(단거리 범퍼 ↔ 더 멀리):
#     ros2 launch humanoid_perception depth_obstacle.launch.py max_range:=0.6
#   pose controller 쪽에서 회피 켜기:
#     ros2 param set /pose_goal_controller_node avoid_enable true
#
#  [확인]
#   - RViz: /depth_obstacle/markers (코리도·검출점·최근접) , /obstacle/avoid_hint(topic echo)
#   - 전방 50cm 안에 물체 두면 hint.x=거리, hint.y=±방향, 빼면 99.0/0
# =============================================================================

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    max_range = LaunchConfiguration('max_range')
    corridor_half = LaunchConfiguration('corridor_half')

    return LaunchDescription([
        DeclareLaunchArgument('max_range', default_value='0.5',
                              description='전방 처리 최대거리[m] (단거리 범퍼 기본 0.5)'),
        DeclareLaunchArgument('corridor_half', default_value='0.25',
                              description='코리도 반폭[m] (|y|≤)'),

        Node(
            package='humanoid_perception',
            executable='depth_obstacle_detector_node',
            name='depth_obstacle_detector_node',
            output='screen',
            parameters=[{
                'max_range': max_range,
                'corridor_half': corridor_half,
                # 단거리에 맞춘 권장값(필요 시 ros2 param set 으로 라이브 튜닝)
                'min_range': 0.1,
                'min_obstacle_height': 0.06,   # 지면 위 이만큼 솟으면 장애물
                'max_obstacle_height': 1.0,
                'ground_bin_m': 0.1,           # 단거리니 구간 촘촘히
                'ground_percentile': 20.0,
                'min_cluster_points': 8,       # 외딴 튐 제거
                'hint_topic': '/obstacle/avoid_hint',
                'hint_center_deadband': 0.05,
                'hint_default_dir': 1.0,
            }],
        ),
    ])
