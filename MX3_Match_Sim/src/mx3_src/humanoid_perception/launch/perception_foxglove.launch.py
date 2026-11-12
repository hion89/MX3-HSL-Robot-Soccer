# perception_foxglove.launch.py
"""
RealSense + YOLO + image_compress + foxglove_bridge 통합 launch.

기존 perception_clean.launch.py 의 모든 기능 +
PC 의 Foxglove Studio 에서 실시간으로 볼 수 있게 stream.

실행:
  ros2 launch humanoid_perception perception_foxglove.launch.py

PC 의 Foxglove Studio 에서 ws://<jetson-ip>:8765 로 연결.

부팅 순서:
  0초:    RealSense launch 시작 (별도 프로세스)
  3초:    YOLO detector 시작 (카메라 안정화 후)
  8초:    image_transport republish 시작 (raw → JPEG 압축)
  8초:    foxglove_bridge 시작 (PC 연결 대기)

옵션 예:
  ros2 launch humanoid_perception perception_foxglove.launch.py \\
      conf:=0.4 \\
      foxglove_port:=8765
"""
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    TimerAction,
    OpaqueFunction,
    IncludeLaunchDescription,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def _setup_extra(context, *args, **kwargs):
    """
    perception_clean.launch.py 가 띄운 YOLO 위에 추가로 띄울 노드들:
      - image_transport republish: raw → compressed (PC 대역폭 1/25)
      - foxglove_bridge: 화이트리스트로 압축 토픽 + 검출만 stream
    """
    foxglove_port = LaunchConfiguration('foxglove_port').perform(context)

    # ── 1) raw → compressed 변환 ──
    # 토픽 이름: /perception/annotated → /perception/annotated_repub/compressed
    image_repub = Node(
        package='image_transport',
        executable='republish',
        name='image_republisher',
        arguments=['raw', 'compressed'],
        remappings=[
            ('/in',              '/perception/annotated'),
            ('/out/compressed',  '/perception/annotated_repub/compressed'),
        ],
        output='screen',
    )

    # ── 2) foxglove_bridge — 화이트리스트로 제한 ──
    # 이미지 raw 토픽은 절대 stream 안 함 (메모리/네트워크 보호)
    foxglove = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        parameters=[{
            'port': int(foxglove_port),
            'topic_whitelist': [
                # 우리 perception 데이터
                '/perception/annotated_repub/compressed',
                '/perception/detections',
                # 추후 다른 노드 추가 시 필요한 항목 화이트리스트에 추가
                # '/joint_states',
                # '/imu/data',
                # '/board/.*',
            ],
            'send_buffer_limit': 10_000_000,
            'use_compression': True,
        }],
        output='screen',
    )

    # YOLO 가 살아있는 토픽 (/perception/annotated) 을 만든 뒤에 시작해야 함.
    # perception_clean.launch.py 에서 YOLO 는 ~3초 후 시작 → 여기는 그 +5초 = 8초.
    return [
        TimerAction(period=8.0, actions=[image_repub, foxglove]),
    ]


def generate_launch_description():
    pkg_share = get_package_share_directory('humanoid_perception')
    perception_launch = os.path.join(
        pkg_share, 'launch', 'perception_clean.launch.py')

    return LaunchDescription([
        # 통합 launch 만의 인자
        DeclareLaunchArgument(
            'foxglove_port', default_value='8765',
            description='Foxglove WebSocket 서버 포트'),

        # 기존 perception_clean.launch.py 그대로 include
        # (RealSense + YOLO 자동 시작)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(perception_launch),
        ),

        # 추가로 띄울 것들 (8초 후 지연)
        OpaqueFunction(function=_setup_extra),
    ])