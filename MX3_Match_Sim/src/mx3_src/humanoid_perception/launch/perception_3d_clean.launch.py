# perception_3d_clean.launch.py
"""
RealSense D405 + YOLO 3D Cartesian 객체 검출 통합 launch (경고 없는 버전).

`perception_clean.launch.py` 의 3D 버전 — 기존 2D 노드 대신
`yolo_detector_3d_node` 를 실행하여 각 검출의 (X, Y, Z) 를 카메라 광학 좌표계 기준으로 publish 한다.

기존 perception_clean.launch.py 와 동일하게:
  - RealSense launch 를 IncludeLaunchDescription 대신 ExecuteProcess 로 실행
    → 부모 launch arg 가 RealSense 로 새지 않아 'Parameter X is not supported' 경고 0
  - 부팅 순서:
      1. 즉시 RealSense launch 시작 (ros2 launch realsense2_camera rs_launch.py)
      2. yolo_delay_sec 초 후 YOLO 3D detector 시작
         (RealSense 의 컬러/깊이/CameraInfo 가 안정적으로 발행되기를 기다림)

차이점:
  - YOLO 노드를 `yolo_detector_3d_node` 로 변경
  - `camera_info_topic`, `output_frame_id`, `depth_patch_radius`, `drop_invalid_depth`
    파라미터를 추가로 전달
  - max_depth_m 기본값을 3.0 으로 축소 (D405 권장 작동 범위)

기본 실행:
  ros2 launch humanoid_perception perception_3d_clean.launch.py

옵션:
  ros2 launch humanoid_perception perception_3d_clean.launch.py \\
      engine_path:=/some/path.engine \\
      conf:=0.4 \\
      yolo_delay_sec:=5.0 \\
      output_frame_id:=camera_color_optical_frame \\
      drop_invalid_depth:=true

출력 토픽:
  /perception/detections_3d  (vision_msgs/Detection3DArray)
  /perception/annotated_3d   (sensor_msgs/Image)
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


def _yolo3d_setup(context, *args, **kwargs):
    """
    OpaqueFunction: launch arg 를 평가한 뒤 YOLO 3D Node 생성.

    OpaqueFunction 을 쓰는 이유:
      - 'true'/'false' 문자열 → bool, '0.4' → float 같은 타입 변환을
        파이썬 코드로 직접 해야 정확하다.
      - LaunchConfiguration 객체를 Node parameters 에 그대로 넘기면 모두 문자열이라
        rclpy declare_parameter 의 타입 추론이 꼬일 수 있음.
    """
    # ── 추론/모델 파라미터 ─────────────────────────────────
    engine_path = LaunchConfiguration('engine_path').perform(context)
    conf        = LaunchConfiguration('conf').perform(context)
    iou         = LaunchConfiguration('iou').perform(context)
    imgsz       = LaunchConfiguration('imgsz').perform(context)
    annotated   = LaunchConfiguration('annotated').perform(context)

    # ── 토픽 / Frame 파라미터 ─────────────────────────────
    color_topic       = LaunchConfiguration('color_topic').perform(context)
    depth_topic       = LaunchConfiguration('depth_topic').perform(context)
    camera_info_topic = LaunchConfiguration('camera_info_topic').perform(context)
    output_frame_id   = LaunchConfiguration('output_frame_id').perform(context)

    # ── 깊이 처리 파라미터 ────────────────────────────────
    min_depth_m         = LaunchConfiguration('min_depth_m').perform(context)
    max_depth_m         = LaunchConfiguration('max_depth_m').perform(context)
    depth_patch_radius  = LaunchConfiguration('depth_patch_radius').perform(context)
    drop_invalid_depth  = LaunchConfiguration('drop_invalid_depth').perform(context)

    # ── 부팅 지연 ─────────────────────────────────────────
    yolo_delay = float(LaunchConfiguration('yolo_delay_sec').perform(context))

    # ── 3D YOLO 노드 정의 ─────────────────────────────────
    yolo3d_node = Node(
        package='humanoid_perception',
        executable='yolo_detector_3d_node',   # setup.py 의 console_scripts 에 등록돼 있어야 함
        name='yolo_detector_3d',
        output='screen',
        parameters=[{
            # 모델
            'engine_path':         engine_path,
            'conf_threshold':      float(conf),
            'iou_threshold':       float(iou),
            'imgsz':               int(imgsz),
            'publish_annotated':   annotated.lower() == 'true',

            # 토픽 / Frame
            'color_topic':         color_topic,
            'depth_topic':         depth_topic,
            'camera_info_topic':   camera_info_topic,
            'output_frame_id':     output_frame_id,

            # 깊이 처리
            'min_depth_m':         float(min_depth_m),
            'max_depth_m':         float(max_depth_m),
            'depth_patch_radius':  int(depth_patch_radius),
            'drop_invalid_depth':  drop_invalid_depth.lower() == 'true',

            # 학습 모델 클래스 ID 매핑 (best.pt 의 names 와 동일해야 함)
            #   0=ball, 1=goalpost, 2=robot, 3=field_edge, 4=line,
            #   5=X-Intersection, 6=T-Intersection, 7=L-Intersection, 8=obstacle
            'class_names': ['ball', 'goalpost', 'robot',
                            'field_edge', 'line',
                            'X-Intersection', 'T-Intersection',
                            'L-Intersection', 'obstacle'],
        }],
    )

    # RealSense 가 첫 프레임 + CameraInfo 를 publish 하기까지 시간이 걸림 →
    # 약간 지연 후 YOLO 3D 시작. (특히 CameraInfo 가 늦으면 첫 몇 프레임은 X/Y 가 0 으로 publish 됨)
    return [TimerAction(period=yolo_delay, actions=[yolo3d_node])]


def generate_launch_description():
    # 패키지의 share 디렉토리 안 models/best.engine 을 기본 엔진으로 사용.
    default_engine = os.path.join(
        get_package_share_directory('humanoid_perception'),
        'models', 'best.engine')

    # ── RealSense 를 ExecuteProcess 로 띄움 ──
    # 'ros2 launch realsense2_camera rs_launch.py ...' 를 그대로 자식 프로세스로 실행.
    # 이렇게 하면 부모 launch arg 가 자동 전달되지 않아서 RealSense launch 가
    # 모르는 인자를 만났을 때 발생하던 'Parameter X is not supported' 경고가 사라진다.
    realsense_proc = ExecuteProcess(
        cmd=[
            'ros2', 'launch', 'realsense2_camera', 'rs_launch.py',
            'device_type:=d405',
            'enable_color:=true',
            'enable_depth:=true',
            # 컬러/깊이 프로파일을 동일하게 — align_depth 시 픽셀 1:1 정렬을 안정화
            ['rgb_camera.color_profile:=', LaunchConfiguration('rs_camera_profile')],
            ['depth_module.depth_profile:=', LaunchConfiguration('rs_camera_profile')],
            # ★ 가장 중요: depth 를 color frame 에 정렬.
            # 이 옵션이 꺼져있으면 yolo_detector_3d_node 의 (u,v) → (X,Y,Z) 역투영이 어긋난다.
            'align_depth.enable:=true',
            # 점군은 끔 — YOLO 추론과 동시에 켜면 Jetson CPU/GPU 부하 큼
            'pointcloud.enable:=false',
        ],
        output='screen',
    )

    return LaunchDescription([
        # ──────────────────────────────────────────────────
        # YOLO 모델 / 추론 파라미터
        # ──────────────────────────────────────────────────
        DeclareLaunchArgument('engine_path', default_value=default_engine,
            description='TensorRT engine 파일 경로 (.engine)'),
        DeclareLaunchArgument('conf',        default_value='0.25',
            description='검출 confidence threshold'),
        DeclareLaunchArgument('iou',         default_value='0.45',
            description='NMS IoU threshold'),
        DeclareLaunchArgument('imgsz',       default_value='640',
            description='추론 입력 해상도 (engine 빌드 입력 크기와 일치 권장)'),
        DeclareLaunchArgument('annotated',   default_value='true',
            description='/perception/annotated_3d 시각화 발행 여부'),

        # ──────────────────────────────────────────────────
        # 토픽 / Frame 파라미터
        # ──────────────────────────────────────────────────
        DeclareLaunchArgument('color_topic',
            default_value='/camera/camera/color/image_raw',
            description='RealSense 컬러 이미지 토픽'),
        DeclareLaunchArgument('depth_topic',
            default_value='/camera/camera/aligned_depth_to_color/image_raw',
            description='aligned depth 이미지 토픽 (color 와 픽셀 정렬됨)'),
        DeclareLaunchArgument('camera_info_topic',
            default_value='/camera/camera/color/camera_info',
            description='CameraInfo 토픽 (color intrinsics — aligned depth 에도 동일 적용)'),
        DeclareLaunchArgument('output_frame_id',
            default_value='camera_color_optical_frame',
            description='Detection3DArray.header.frame_id 로 사용할 frame. '
                        '빈 문자열이면 color 이미지 헤더의 frame_id 그대로 사용.'),

        # ──────────────────────────────────────────────────
        # 깊이 처리 파라미터
        # ──────────────────────────────────────────────────
        DeclareLaunchArgument('min_depth_m', default_value='0.07',
            description='유효 거리 하한(m) — D405 MinZ'),
        DeclareLaunchArgument('max_depth_m', default_value='3.0',
            description='유효 거리 상한(m) — D405 는 ~1m 권장, 여유 두고 3m'),
        DeclareLaunchArgument('depth_patch_radius', default_value='3',
            description='bbox 중심에서 깊이 추출용 patch 반경(px). 3 → 7x7 patch'),
        DeclareLaunchArgument('drop_invalid_depth', default_value='true',
            description='true(기본): 깊이 측정 실패한 detection 을 결과에서 제외. '
                        'false: X=Y=Z=0 으로 publish — 소비자가 이를 실제 좌표로 '
                        '오인하는 문제가 있다. yolo_3d.launch.py 와 같은 기본값 유지'),

        # ──────────────────────────────────────────────────
        # RealSense 파라미터
        # ──────────────────────────────────────────────────
        DeclareLaunchArgument('rs_camera_profile', default_value='848x480x30',
            description='RealSense color + depth 프로파일 (WxHxFPS). '
                        'D405 컬러는 848x480 또는 1280x720 지원'),

        # ──────────────────────────────────────────────────
        # 부팅 순서 제어
        # ──────────────────────────────────────────────────
        DeclareLaunchArgument('yolo_delay_sec', default_value='1.0',
            description='RealSense 시작 후 YOLO 3D 시작까지 대기 시간 (초). '
                        '카메라 첫 프레임 + CameraInfo 가 안정화될 때까지 기다림.'),

        # ──────────────────────────────────────────────────
        # 실행 — RealSense 먼저 띄우고 OpaqueFunction 안에서 TimerAction 으로 YOLO 지연 시작
        # ──────────────────────────────────────────────────
        realsense_proc,
        OpaqueFunction(function=_yolo3d_setup),
    ])