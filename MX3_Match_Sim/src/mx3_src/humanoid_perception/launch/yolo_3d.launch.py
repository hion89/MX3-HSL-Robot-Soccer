# yolo_3d.launch.py
"""
YOLO 3D 검출 노드 (yolo_detector_3d_node) 만 띄우는 launch.
RealSense 카메라는 별도 터미널에서 미리 띄워둬야 함.

이 launch 는 기존 `yolo_only.launch.py` 와 완전히 독립적이다.
같은 RealSense 카메라 입력을 공유하면서 다른 토픽으로 결과를 publish 하므로
두 노드를 동시에 띄워도 충돌 없음 (필요 시 GPU 메모리/추론 부하만 주의).

사용 패턴:
  # 터미널 1 — 카메라
  # align_depth.enable:=true 가 반드시 필요. depth 가 color frame 으로 정렬되어야
  # color intrinsics 로 depth 픽셀을 역투영할 수 있다.
  ros2 launch realsense2_camera rs_launch.py \\
      device_type:=d405 align_depth.enable:=true \\
      rgb_camera.color_profile:=848x480x30 \\
      depth_module.depth_profile:=848x480x30

  # 터미널 2 — YOLO 3D
  ros2 launch humanoid_perception yolo_3d.launch.py

옵션 예:
  # 신뢰도 임계값 0.4, output frame 명시
  ros2 launch humanoid_perception yolo_3d.launch.py \\
      conf:=0.4 output_frame_id:=camera_color_optical_frame

  # 다른 engine 사용
  ros2 launch humanoid_perception yolo_3d.launch.py \\
      engine_path:=/home/khuber/models/best_v2.engine

  # 깊이 측정 실패한 detection 은 결과에서 제외 (2D fallback 안 받고 싶을 때)
  ros2 launch humanoid_perception yolo_3d.launch.py \\
      drop_invalid_depth:=true

출력 토픽:
  /perception/detections_3d  (vision_msgs/Detection3DArray) — 메인 산출물
  /perception/annotated_3d   (sensor_msgs/Image)            — 시각화 (옵션)
"""
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # 패키지의 share 디렉토리 안 models/best.engine 을 기본 엔진으로 사용.
    # setup.py 의 data_files 가 share/<pkg>/models/*.engine 을 install 함.
    default_engine = os.path.join(
        get_package_share_directory('humanoid_perception'),
        'models', 'best.engine')

    return LaunchDescription([
        # ── 모델/추론 파라미터 ───────────────────────────────
        DeclareLaunchArgument('engine_path', default_value=default_engine,
            description='TensorRT engine 파일 절대경로 (.engine)'),
        DeclareLaunchArgument('conf',        default_value='0.25',
            description='YOLO confidence threshold'),
        DeclareLaunchArgument('iou',         default_value='0.45',
            description='YOLO NMS IoU threshold'),
        DeclareLaunchArgument('imgsz',       default_value='640',
            description='추론 입력 해상도 (engine 빌드 시점 입력 크기와 일치 권장)'),
        DeclareLaunchArgument('annotated',   default_value='true',
            description='/perception/annotated_3d 시각화 이미지 publish 여부'),

        # ── 토픽 / Frame 파라미터 ────────────────────────────
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

        # ── 깊이 처리 파라미터 ──────────────────────────────
        DeclareLaunchArgument('min_depth_m', default_value='0.07',
            description='유효 거리 하한(m) — D405 MinZ'),
        DeclareLaunchArgument('max_depth_m', default_value='3.0',
            description='유효 거리 상한(m) — D405 는 ~1m 권장, 여유 두고 3m'),
        DeclareLaunchArgument('depth_patch_radius', default_value='3',
            description='bbox 중심에서 깊이 추출용 patch 반경(px). 3 → 7x7 patch'),
        DeclareLaunchArgument('drop_invalid_depth', default_value='true',
            description='true(기본): 깊이 측정 실패한 detection 을 결과에서 제외. '
                        'false: 2D 박스만 있는 detection 도 publish (X=Y=Z=0) — '
                        '★ (0,0,0) 은 "깊이 미상" 이라는 뜻인데 메시지엔 그 구분이 '
                        '없어서, min_valid_z 게이트가 없는 소비자'
                        '(visual_servo_kick_node)가 실제 좌표로 읽고 오동작한다. '
                        '깊이 파이프라인 자체를 의심할 때만 false 로 내려 볼 것'),
        DeclareLaunchArgument('ball_min_dist_m', default_value='0.30',
            description='ball 근거리 게이트. 카메라~공 직선거리(Z)가 이 값보다 '
                        '가까운 ball 검출은 로봇 자기 몸 오검출로 보고 무시. '
                        '자기 몸 오검출이 남으면 키우고, 발 앞 공이 사라지면 줄인다.'),

        # ── 노드 정의 ───────────────────────────────────────
        Node(
            package='humanoid_perception',
            executable='yolo_detector_3d_node',  # setup.py 의 console_scripts 에 등록 필요
            name='yolo_detector_3d',
            output='screen',
            parameters=[{
                'engine_path':         LaunchConfiguration('engine_path'),
                'conf_threshold':      LaunchConfiguration('conf'),
                'iou_threshold':       LaunchConfiguration('iou'),
                'imgsz':               LaunchConfiguration('imgsz'),
                'publish_annotated':   LaunchConfiguration('annotated'),

                'color_topic':         LaunchConfiguration('color_topic'),
                'depth_topic':         LaunchConfiguration('depth_topic'),
                'camera_info_topic':   LaunchConfiguration('camera_info_topic'),
                'output_frame_id':     LaunchConfiguration('output_frame_id'),

                'min_depth_m':         LaunchConfiguration('min_depth_m'),
                'max_depth_m':         LaunchConfiguration('max_depth_m'),
                'depth_patch_radius':  LaunchConfiguration('depth_patch_radius'),
                'drop_invalid_depth':  LaunchConfiguration('drop_invalid_depth'),
                'ball_min_dist_m':     LaunchConfiguration('ball_min_dist_m'),

                # 학습된 .pt 의 m.names 와 동일해야 함 — class_id → label 매핑
                'class_names': ['ball', 'goalpost', 'robot',
                                'field_edge', 'line',
                                'X-Intersection', 'T-Intersection',
                                'L-Intersection', 'obstacle'],
            }],
        ),
    ])