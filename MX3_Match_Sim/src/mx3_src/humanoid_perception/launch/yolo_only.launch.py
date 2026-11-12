# yolo_only.launch.py
"""
YOLO 검출 노드만 띄우는 launch.
RealSense 카메라는 별도로 미리 띄워둬야 함.

사용 패턴:
  # 터미널 1 — 카메라
  ros2 launch realsense2_camera rs_launch.py \\
      device_type:=d405 align_depth.enable:=true \\
      rgb_camera.color_profile:=848x480x30 \\
      depth_module.depth_profile:=848x480x30

  # 터미널 2 — YOLO
  ros2 launch humanoid_perception yolo_only.launch.py

옵션 예:
  ros2 launch humanoid_perception yolo_only.launch.py \\
      engine_path:=/some/path.engine conf:=0.4
"""
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    default_engine = os.path.join(
        get_package_share_directory('humanoid_perception'),
        'models', 'best.engine')

    return LaunchDescription([
        DeclareLaunchArgument('engine_path', default_value=default_engine),
        DeclareLaunchArgument('conf',        default_value='0.25'),
        DeclareLaunchArgument('iou',         default_value='0.45'),
        DeclareLaunchArgument('imgsz',       default_value='640'),
        DeclareLaunchArgument('annotated',   default_value='true'),
        DeclareLaunchArgument('use_depth',   default_value='true'),
        DeclareLaunchArgument('min_depth_m', default_value='0.07'),
        DeclareLaunchArgument('max_depth_m', default_value='20.0'),

        Node(
            package='humanoid_perception',
            executable='yolo_detector_node',
            name='yolo_detector',
            output='screen',
            parameters=[{
                'engine_path':       LaunchConfiguration('engine_path'),
                'conf_threshold':    LaunchConfiguration('conf'),
                'iou_threshold':     LaunchConfiguration('iou'),
                'imgsz':             LaunchConfiguration('imgsz'),
                'publish_annotated': LaunchConfiguration('annotated'),
                'use_depth':         LaunchConfiguration('use_depth'),
                'min_depth_m':       LaunchConfiguration('min_depth_m'),
                'max_depth_m':       LaunchConfiguration('max_depth_m'),
                'color_topic':       '/camera/camera/color/image_raw',
                'depth_topic':       '/camera/camera/aligned_depth_to_color/image_raw',
                'class_names': ['ball', 'goalpost', 'robot',
                                'field_edge', 'line',
                                'X-Intersection', 'T-Intersection',
                                'L-Intersection', 'obstacle'],
            }],
        ),
    ])