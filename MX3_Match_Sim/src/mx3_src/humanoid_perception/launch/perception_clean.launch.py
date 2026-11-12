# perception_clean.launch.py
"""
RealSense D405 + YOLO 객체 검출 통합 launch (경고 없는 버전).

기존 perception.launch.py 와 동일한 기능이지만,
RealSense launch 를 IncludeLaunchDescription 대신
ExecuteProcess 로 별도 프로세스 실행 → 부모 launch arg 가 RealSense 로
새지 않아서 'Parameter X is not supported' 경고가 발생하지 않는다.

부팅 순서:
  1. 즉시 RealSense launch 시작 (ros2 launch realsense2_camera rs_launch.py)
  2. 3초 후 YOLO detector 시작
     (RealSense 첫 프레임이 안정적으로 발행되기까지 기다림)

기본 실행:
  ros2 launch humanoid_perception perception_clean.launch.py

옵션:
  ros2 launch humanoid_perception perception_clean.launch.py \\
      engine_path:=/some/path.engine \\
      conf:=0.4 \\
      yolo_delay_sec:=5.0
"""
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    TimerAction,
    OpaqueFunction,
)
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def _yolo_setup(context, *args, **kwargs):
    """OpaqueFunction: launch arg 평가 후 YOLO Node 생성."""
    engine_path = LaunchConfiguration('engine_path').perform(context)
    conf        = LaunchConfiguration('conf').perform(context)
    iou         = LaunchConfiguration('iou').perform(context)
    imgsz       = LaunchConfiguration('imgsz').perform(context)
    annotated   = LaunchConfiguration('annotated').perform(context)
    use_depth   = LaunchConfiguration('use_depth').perform(context)
    min_depth_m = LaunchConfiguration('min_depth_m').perform(context)
    max_depth_m = LaunchConfiguration('max_depth_m').perform(context)
    yolo_delay  = float(LaunchConfiguration('yolo_delay_sec').perform(context))

    yolo_node = Node(
        package='humanoid_perception',
        executable='yolo_detector_node',
        name='yolo_detector',
        output='screen',
        parameters=[{
            'engine_path':       engine_path,
            'conf_threshold':    float(conf),
            'iou_threshold':     float(iou),
            'imgsz':             int(imgsz),
            'publish_annotated': annotated.lower() == 'true',
            'use_depth':         use_depth.lower() == 'true',
            'min_depth_m':       float(min_depth_m),
            'max_depth_m':       float(max_depth_m),
            'color_topic':       '/camera/camera/color/image_raw',
            'depth_topic':       '/camera/camera/aligned_depth_to_color/image_raw',
            # 학습 모델 클래스 ID 매핑 (best.pt 의 names 와 동일)
            #   0=ball, 1=goalpost, 2=robot, 3=field_edge, 4=line,
            #   5=X-Intersection, 6=T-Intersection, 7=L-Intersection, 8=obstacle
            'class_names': ['ball', 'goalpost', 'robot',
                            'field_edge', 'line',
                            'X-Intersection', 'T-Intersection',
                            'L-Intersection', 'obstacle'],
        }],
    )

    # RealSense 가 first frame publish 까지 시간 걸림 → 약간 지연 후 YOLO 시작
    return [TimerAction(period=yolo_delay, actions=[yolo_node])]


def generate_launch_description():
    default_engine = os.path.join(
        get_package_share_directory('humanoid_perception'),
        'models', 'best.engine')

    # ── RealSense 를 ExecuteProcess 로 띄움 ──
    # 일반 'ros2 launch ...' 명령을 그대로 부 process 로 실행.
    # 이렇게 하면 부모 launch arg 가 자동 전달되지 않아 경고 0.
    realsense_proc = ExecuteProcess(
        cmd=[
            'ros2', 'launch', 'realsense2_camera', 'rs_launch.py',
            'device_type:=d405',
            'enable_color:=true',
            'enable_depth:=true',
            ['rgb_camera.color_profile:=', LaunchConfiguration('rs_camera_profile')],
            ['depth_module.depth_profile:=', LaunchConfiguration('rs_camera_profile')],
            'align_depth.enable:=true',
            'pointcloud.enable:=false',
        ],
        output='screen',
    )

    return LaunchDescription([
        # ── YOLO 파라미터 ──
        DeclareLaunchArgument('engine_path', default_value=default_engine,
            description='TensorRT engine 파일 경로 (.engine)'),
        DeclareLaunchArgument('conf',        default_value='0.25',
            description='검출 confidence threshold'),
        DeclareLaunchArgument('iou',         default_value='0.45',
            description='NMS IoU threshold'),
        DeclareLaunchArgument('imgsz',       default_value='640',
            description='추론 입력 해상도'),
        DeclareLaunchArgument('annotated',   default_value='true',
            description='/perception/annotated 시각화 발행 여부'),
        DeclareLaunchArgument('use_depth',   default_value='true',
            description='aligned depth 구독해서 거리 첨부'),
        DeclareLaunchArgument('min_depth_m', default_value='0.07',
            description='유효 거리 하한'),
        DeclareLaunchArgument('max_depth_m', default_value='20.0',
            description='유효 거리 상한'),

        # ── RealSense 파라미터 ──
        DeclareLaunchArgument('rs_camera_profile', default_value='848x480x30',
            description='RealSense color + depth 프로파일 (WxHxFPS). '
                        'D405 컬러는 848x480 또는 1280x720 지원'),

        # ── 부팅 순서 제어 ──
        DeclareLaunchArgument('yolo_delay_sec', default_value='1.0',
            description='RealSense 시작 후 YOLO 시작까지 대기 시간 (초). '
                        '카메라 첫 프레임이 안정화될 때까지 기다림.'),

        # ── 실행 ──
        realsense_proc,
        OpaqueFunction(function=_yolo_setup),
    ])