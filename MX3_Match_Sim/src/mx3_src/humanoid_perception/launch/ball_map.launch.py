# -*- coding: utf-8 -*-
# =============================================================================
#  ball_map.launch.py
#
#  공 → map 투영 노드(ball_map_projector_node)를 단독 실행하는 런치.
#
#  [전제]
#   - yolo_detector_3d_node 가 이미 /perception/detections_3d 를 발행 중이어야 함.
#   - 로컬라이제이션 스택이 map→base_link TF 를, robot_state_publisher 가
#     base_link→카메라 광학프레임 TF 를 발행 중이어야 함(체인 완성).
#
#  [실행]
#     ros2 launch humanoid_perception ball_map.launch.py
#   파라미터 덮어쓰기 예:
#     ros2 launch humanoid_perception ball_map.launch.py ball_radius_m:=0.05 max_range_m:=4.0
# =============================================================================

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ── 런치 인자(필요한 것만 노출, 나머지는 노드 기본값 사용) ──
    detections_topic = LaunchConfiguration('detections_topic')
    map_frame = LaunchConfiguration('map_frame')
    ball_class_name = LaunchConfiguration('ball_class_name')
    min_score = LaunchConfiguration('min_score')
    max_range_m = LaunchConfiguration('max_range_m')
    ball_radius_m = LaunchConfiguration('ball_radius_m')
    marker_diameter_m = LaunchConfiguration('marker_diameter_m')
    project_to_ground = LaunchConfiguration('project_to_ground')

    return LaunchDescription([
        DeclareLaunchArgument('detections_topic',
                              default_value='/perception/detections_3d'),
        DeclareLaunchArgument('map_frame', default_value='map'),
        DeclareLaunchArgument('ball_class_name', default_value='ball'),
        DeclareLaunchArgument('min_score', default_value='0.40'),
        DeclareLaunchArgument('max_range_m', default_value='5.0'),
        DeclareLaunchArgument('ball_radius_m', default_value='0.05'),
        # 0 = 자동(2×ball_radius_m), >0 = 마커 지름 강제(물리 높이와 분리)
        DeclareLaunchArgument('marker_diameter_m', default_value='0.0'),
        DeclareLaunchArgument('project_to_ground', default_value='true'),

        Node(
            package='humanoid_perception',
            executable='ball_map_projector_node',
            name='ball_map_projector_node',
            output='screen',
            parameters=[{
                'detections_topic': detections_topic,
                'map_frame': map_frame,
                'ball_class_name': ball_class_name,
                'min_score': min_score,
                'max_range_m': max_range_m,
                'ball_radius_m': ball_radius_m,
                'marker_diameter_m': marker_diameter_m,
                'project_to_ground': project_to_ground,
                # 나머지(ema_alpha, lost_timeout_s, marker_color_rgba 등)는 노드 기본값 사용.
            }],
        ),
    ])
