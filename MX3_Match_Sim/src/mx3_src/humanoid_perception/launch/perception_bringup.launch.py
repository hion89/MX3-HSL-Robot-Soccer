# perception_bringup.launch.py
"""
KHUBER 인지(perception) 통합 bringup launch.

이 launch 가 하는 일 (한 방에 인지 스택 전체 기동):
  1. robot_state_publisher 실행
       - URDF 를 읽어 TF 를 publish.
       - tf_minimal=true(기본)면 목+카메라 체인만 담은 경량 URDF 를 써서
         base_link → neck_yaw_link → neck_pitch_link → camera_link 만 발행한다.
         (팔/다리 link TF 는 프로젝트에서 아무도 lookup 하지 않으므로 제외 →
          /tf 변환 수 ~20개→2개로 줄여 listener 노드들의 CPU 낭비 제거)
       - tf_minimal=false 면 팔/다리 포함 풀 트리(원본 URDF)를 발행(3D 모델 시각화용).
       - 각 revolute joint 의 실제 각도는 /joint_states 토픽에서 받아야 함.
         (joint_states 가 없으면 모든 joint 를 0 으로 가정 → 머리 돌려도 TF 안 움직임)
  2. RealSense D405 실행 (enable_camera=true)
       - camera_link → camera_color_optical_frame 등 카메라 내부 TF 를 자동 publish.
       - URDF 의 camera_mount_joint(neck_pitch_link → camera_link) 덕분에
         base_link → camera_color_optical_frame 까지 하나의 tree 로 연결된다.
  3. YOLO 3D 검출 (enable_yolo=true)  ★복원★
       - yolo_3d.launch.py 를 include → yolo_detector_3d_node 기동.
       - /perception/detections_3d (vision_msgs/Detection3DArray) 발행.
         공/골포스트/로봇/라인 등 3D 검출. neck_ball_tracker·vision_servo·
         ball_map_projector 의 공통 입력원이다.
       - 카메라(2)가 떠 있어야 동작한다(색/정렬depth/카메라info 토픽 소비).
  4. 공 → map 투영 (enable_ball_map=true)  ★복원★
       - ball_map.launch.py 를 include → ball_map_projector_node 기동.
       - /ball/map_point (map 프레임 공 위치) 발행. ball_follow 의 입력원.
       - (3)의 /perception/detections_3d 를 소비하므로 (3) 뒤에 include 한다.
       - map→base_link TF(로컬라이제이션)와 base_link→카메라 TF(RSP)가 있어야
         투영 체인이 완성된다. 없으면 노드가 조용히 대기(에러 아님).

  ※ 왜 3·4 가 "복원" 인가:
      과거 이 파일은 RSP + 카메라 + YOLO3D + 공투영을 모두 품는 인지 bringup 이었다.
      tf_minimal 최적화 작업 때 TF+카메라 전용으로 덮이면서 3·4 가 빠졌고, 그 결과
      match.launch.py 로 띄워도 YOLO 검출/공투영이 통째로 죽는 문제가 있었다.
      → 여기서 yolo_3d/ball_map 을 include 방식으로 되살린다(파라미터 원본은 각
        leaf 런치에 그대로 두므로 튜닝값이 두 군데로 갈라지지 않는다 = DRY).

토글:
  - enable_camera : RealSense D405 실행 여부(기본 true). false 면 YOLO/공투영도 입력이
                    없으니 함께 끄는 게 맞다(카메라 없이 검출 불가).
  - enable_yolo   : YOLO 3D 검출 노드 실행 여부(기본 true).
  - enable_ball_map: 공 map 투영 노드 실행 여부(기본 true). YOLO 만 쓰고 싶으면 false.

URDF 경로:
  - urdf:= 를 비우면(기본) tf_minimal 에 따라 share/humanoid_perception/urdf/ 의
    XM_robot03_tf_min.urdf(최소) 또는 XM_robot03_output.urdf(풀)를 자동 선택.
    (setup.py 가 urdf/ 를 install 하므로 colcon build 후 사용 가능)
  - 특정 URDF 를 강제하려면 urdf:=/절대/경로.urdf 로 override.

전제 조건:
  - /joint_states 가 어딘가에서 publish 되고 있어야 함
    (목 각도는 neck_state_parser_node 가 neck_yaw_joint/neck_pitch_joint 로 발행.
     joint 이름이 URDF 와 정확히 일치해야 함 — tf_min URDF 도 이 이름을 그대로 유지)
  - 테스트만 할 거라면 joint_state_publisher_gui 로 대체 가능 (use_jsp_gui:=true)

사용:
  # 기본 (RSP + 카메라 + YOLO3D + 공투영 전부 — match 실운용 구성)
  ros2 launch humanoid_perception perception_bringup.launch.py

  # 검출/투영 없이 TF+카메라만 (카메라 화면만 확인할 때)
  ros2 launch humanoid_perception perception_bringup.launch.py \
      enable_yolo:=false enable_ball_map:=false

  # YOLO 검출만, 공 map 투영은 제외
  ros2 launch humanoid_perception perception_bringup.launch.py enable_ball_map:=false

  # 팔/다리 포함 풀 TF 트리 (RViz 3D 로봇 모델 시각화 등)
  ros2 launch humanoid_perception perception_bringup.launch.py tf_minimal:=false

  # joint_states 가 아직 없을 때 (TF 시각 점검용 - 슬라이더로 관절 조작)
  ros2 launch humanoid_perception perception_bringup.launch.py use_jsp_gui:=true

확인:
  ros2 node list | grep -E 'yolo_detector_3d|ball_map_projector'
  ros2 topic hz /perception/detections_3d
  ros2 topic echo /ball/map_point --once
  ros2 run tf2_ros tf2_echo base_link camera_color_optical_frame
"""
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, OpaqueFunction,
    IncludeLaunchDescription,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def _setup(context, *args, **kwargs):
    urdf_arg = LaunchConfiguration('urdf').perform(context)
    tf_minimal = LaunchConfiguration('tf_minimal').perform(context).lower() == 'true'
    use_jsp_gui = LaunchConfiguration('use_jsp_gui').perform(context).lower() == 'true'
    enable_cam = LaunchConfiguration('enable_camera').perform(context).lower() == 'true'
    enable_yolo = LaunchConfiguration('enable_yolo').perform(context).lower() == 'true'
    enable_ball_map = LaunchConfiguration('enable_ball_map').perform(context).lower() == 'true'
    rs_profile = LaunchConfiguration('rs_camera_profile').perform(context)

    # 인지 leaf 런치들이 있는 share/humanoid_perception/launch 디렉토리 경로.
    # yolo_3d / ball_map 을 include 할 때 사용(파라미터 원본은 그 파일들에 그대로 둠).
    pkg_share = get_package_share_directory('humanoid_perception')
    launch_dir = os.path.join(pkg_share, 'launch')

    # ── URDF 경로 결정 ────────────────────────────────────────────────
    # 우선순위:
    #   1) urdf:= 를 명시적으로 주면 무조건 그 파일을 쓴다(수동 override).
    #   2) 비어있으면 tf_minimal 값으로 자동 선택:
    #        tf_minimal=true  → XM_robot03_tf_min.urdf  (목+카메라 체인만; 기본값)
    #        tf_minimal=false → XM_robot03_output.urdf   (팔/다리 포함 풀 트리)
    #
    # 왜 기본이 minimal 인가:
    #   프로젝트의 어떤 노드도 팔/다리 link TF 를 조회하지 않는다(실제 조회
    #   프레임은 map/odom/base_link/camera_color_optical_frame 뿐). 풀 트리는
    #   /tf 를 매 메시지 ~20개 변환으로 부풀려 9개 listener 노드의 CPU 를
    #   낭비시키므로, 실운용(match)에서는 minimal 을 기본으로 둔다.
    #   3D 로봇 모델 시각화가 필요하면 tf_minimal:=false 또는 urdf:= 로 풀 URDF.
    if urdf_arg:
        urdf_path = urdf_arg
    else:
        fname = 'XM_robot03_tf_min.urdf' if tf_minimal else 'XM_robot03_output.urdf'
        urdf_path = os.path.join(pkg_share, 'urdf', fname)

    if not os.path.isfile(urdf_path):
        raise RuntimeError(
            f"URDF 파일을 찾을 수 없습니다: {urdf_path}\n"
            f"  - colcon build 를 했는지 확인 (urdf/ 가 share 로 install 되어야 함)\n"
            f"  - tf_min URDF 를 새로 추가했다면, data_files glob 특성상 새 파일은\n"
            f"    --symlink-install 만으로 안 잡힐 수 있으니\n"
            f"    'colcon build --packages-select humanoid_perception' 로 한 번 빌드\n"
            f"  - 또는 urdf:=/절대/경로.urdf 로 직접 지정")

    # 펼쳐진 .urdf 를 문자열로 읽어 robot_description 으로 전달
    with open(urdf_path, 'r') as f:
        robot_description = f.read()

    actions = []

    # 1. robot_state_publisher
    actions.append(Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'publish_frequency': 50.0,  # 정적 TF 재발행 주기
        }],
    ))

    # 2. joint_states 소스
    if use_jsp_gui:
        # 실로봇 joint_states 없이 슬라이더로 관절 조작 (TF 점검 전용)
        actions.append(Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            output='screen',
        ))
    # else: 실로봇에선 joint_bridge_u2d2_node 가 /joint_states 를 발행한다고 가정

    # 3. RealSense D405
    if enable_cam:
        actions.append(ExecuteProcess(
            cmd=[
                'ros2', 'launch', 'realsense2_camera', 'rs_launch.py',
                'device_type:=d405',
                'enable_color:=true',
                'enable_depth:=true',
                f'rgb_camera.color_profile:={rs_profile}',
                f'depth_module.depth_profile:={rs_profile}',
                'align_depth.enable:=true',
                'pointcloud.enable:=false',
            ],
            output='screen',
        ))

    # 4. YOLO 3D 검출 노드 (복원) — yolo_3d.launch.py include
    #    파라미터(engine/conf/토픽/깊이처리 등)는 yolo_3d.launch.py 의 기본값을 그대로
    #    사용한다(원본을 single source of truth 로 유지). 값을 바꿔야 하면 그 파일을 수정.
    #    카메라(3)가 꺼져 있으면 yolo 는 입력 토픽을 못 받아 무한 대기하므로 경고만 남긴다.
    if enable_yolo:
        if not enable_cam:
            print("[perception_bringup] ⚠ enable_yolo=true 인데 enable_camera=false 입니다. "
                  "YOLO 노드는 카메라 토픽이 없어 검출을 못 합니다.")
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(launch_dir, 'yolo_3d.launch.py')),
        ))

    # 5. 공 → map 투영 노드 (복원) — ball_map.launch.py include
    #    /perception/detections_3d(4의 산출물)를 소비하므로 반드시 4 뒤에 둔다.
    #    파라미터는 ball_map.launch.py 기본값 사용(원본 유지).
    if enable_ball_map:
        if not enable_yolo:
            print("[perception_bringup] ⚠ enable_ball_map=true 인데 enable_yolo=false 입니다. "
                  "검출(/perception/detections_3d)이 없어 공 투영이 안 됩니다.")
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(launch_dir, 'ball_map.launch.py')),
        ))

    return actions


def generate_launch_description():
    # URDF 경로/토글은 _setup 에서 평가한다. 여기서는 인자만 선언한다.
    return LaunchDescription([
        DeclareLaunchArgument(
            'urdf', default_value='',
            description='펼쳐진 .urdf 파일 경로(절대경로). 비우면 tf_minimal 에 따라 '
                        '패키지 share 안의 URDF 를 자동 선택. 주면 그 파일을 강제 사용'),
        DeclareLaunchArgument(
            'tf_minimal', default_value='true',
            description='true(기본): 목+카메라 체인만 담은 XM_robot03_tf_min.urdf 사용 → '
                        '/tf 변환 수 ~20개→2개로 축소(매치 실운용 권장). '
                        'false: 팔/다리 포함 풀 URDF(XM_robot03_output.urdf) — 3D 모델 시각화용'),
        DeclareLaunchArgument(
            'use_jsp_gui', default_value='false',
            description='true 면 joint_state_publisher_gui 로 관절 수동 조작 (TF 점검용)'),
        DeclareLaunchArgument(
            'enable_camera', default_value='true',
            description='RealSense D405 실행 여부'),
        DeclareLaunchArgument(
            'enable_yolo', default_value='true',
            description='YOLO 3D 검출 노드(yolo_detector_3d_node) 실행 여부. '
                        'yolo_3d.launch.py 를 include (카메라 필요)'),
        DeclareLaunchArgument(
            'enable_ball_map', default_value='true',
            description='공 map 투영 노드(ball_map_projector_node) 실행 여부. '
                        'ball_map.launch.py 를 include (YOLO 검출 필요)'),
        DeclareLaunchArgument(
            'rs_camera_profile', default_value='848x480x30',
            description='RealSense color+depth 프로파일 (WxHxFPS)'),

        OpaqueFunction(function=_setup),
    ])
