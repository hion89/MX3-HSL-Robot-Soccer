# perception.launch.py
"""
RealSense D405 + YOLO 객체 검출 통합 launch.

띄우는 노드:
  - realsense2_camera: D405 카메라 (color + depth aligned)
  - yolo_detector_node: TensorRT engine 추론 + Detection2DArray 발행

기본 실행 (best.engine 자동 탐색):
  ros2 launch humanoid_perception perception.launch.py

옵션 변경 예:
  ros2 launch humanoid_perception perception.launch.py \\
      engine_path:=/some/other/path.engine \\
      conf:=0.4
"""
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def _launch_setup(context, *args, **kwargs):
    """
    OpaqueFunction 사용: launch arg 들을 실제 값으로 평가해서
    RealSense include 에 우리 인자가 새지 않도록 한다.
    (그냥 IncludeLaunchDescription 만 쓰면 부모 launch arg 전부가
     자식한테 자동 전파되어 'Parameter X is not supported' 경고가 뜬다.)
    """
    # ── 평가된 값들 ──
    engine_path     = LaunchConfiguration('engine_path').perform(context)
    conf_threshold  = LaunchConfiguration('conf').perform(context)
    iou_threshold   = LaunchConfiguration('iou').perform(context)
    imgsz           = LaunchConfiguration('imgsz').perform(context)
    annotated       = LaunchConfiguration('annotated').perform(context)
    use_depth       = LaunchConfiguration('use_depth').perform(context)
    min_depth_m     = LaunchConfiguration('min_depth_m').perform(context)
    max_depth_m     = LaunchConfiguration('max_depth_m').perform(context)
    rs_profile      = LaunchConfiguration('rs_camera_profile').perform(context)

    # ── RealSense launch include ──
    # launch_arguments 에 명시한 RealSense 가 아는 것만 전달.
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('realsense2_camera'),
                'launch', 'rs_launch.py')
        ),
        launch_arguments={
            'device_type':                 'd405',
            'enable_color':                'true',
            'enable_depth':                'true',
            'rgb_camera.color_profile':    rs_profile,
            'depth_module.depth_profile':  rs_profile,
            'align_depth.enable':          'true',
            'pointcloud.enable':           'false',
        }.items(),
    )

    # ── YOLO detector ──
    yolo_node = Node(
        package='humanoid_perception',
        executable='yolo_detector_node',
        name='yolo_detector',
        output='screen',
        parameters=[{
            'engine_path':       engine_path,
            'conf_threshold':    float(conf_threshold),
            'iou_threshold':     float(iou_threshold),
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

    return [realsense_launch, yolo_node]


def generate_launch_description():
    # 기본 engine 경로: 패키지 share/humanoid_perception/models/best.engine
    default_engine = os.path.join(
        get_package_share_directory('humanoid_perception'),
        'models', 'best.engine')

    return LaunchDescription([
        DeclareLaunchArgument(
            'engine_path', default_value=default_engine,
            description='TensorRT engine 파일 경로 (.engine)'),
        DeclareLaunchArgument(
            'conf', default_value='0.25',
            description='검출 confidence threshold'),
        DeclareLaunchArgument(
            'iou', default_value='0.45',
            description='NMS IoU threshold'),
        DeclareLaunchArgument(
            'imgsz', default_value='640',
            description='추론 입력 해상도'),
        DeclareLaunchArgument(
            'annotated', default_value='true',
            description='/perception/annotated 시각화 발행 여부'),
        DeclareLaunchArgument(
            'use_depth', default_value='true',
            description='aligned depth 구독해서 거리 첨부'),
        DeclareLaunchArgument(
            'min_depth_m', default_value='0.07',
            description='유효 거리 하한'),
        DeclareLaunchArgument(
            'max_depth_m', default_value='20.0',
            description='유효 거리 상한'),
        DeclareLaunchArgument(
            'rs_camera_profile', default_value='848x480x30',
            description='RealSense color + depth 프로파일 (WxHxFPS). '
                        'D405 컬러는 848x480 또는 1280x720 지원'),

        OpaqueFunction(function=_launch_setup),
    ])
