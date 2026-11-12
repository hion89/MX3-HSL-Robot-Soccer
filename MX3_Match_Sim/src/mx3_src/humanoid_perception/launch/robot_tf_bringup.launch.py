# robot_tf_bringup.launch.py
"""
KHUBER TF tree 통합 launch.

이 launch 가 하는 일:
  1. robot_state_publisher 실행
       - URDF 를 읽어 base_link ~ 모든 link (20 joint + camera_link) 의 TF 를 publish.
       - 각 revolute joint 의 실제 각도는 /joint_states 토픽에서 받아야 함.
         (joint_states 가 없으면 모든 joint 를 0 으로 가정 → 머리 돌려도 TF 안 움직임)
  2. RealSense D405 실행
       - camera_link → camera_color_optical_frame 등 카메라 내부 TF 를 자동 publish.
       - URDF 의 camera_mount_joint(neck_pitch_link → camera_link) 덕분에
         base_link → camera_color_optical_frame 까지 하나의 tree 로 연결된다.

URDF 경로:
  - 기본값으로 이 패키지의 share/humanoid_perception/urdf/XM_robot03_output.urdf 를
    자동으로 잡는다. (setup.py 가 urdf/ 를 install 하므로 colcon build 후 사용 가능)
  - 다른 URDF 를 쓰려면 urdf:=/절대/경로.urdf 로 override.

전제 조건:
  - /joint_states 가 어딘가에서 publish 되고 있어야 함
    (humanoid_serial_bridge 의 joint_bridge_u2d2_node 등에서 18 다리/팔 + 2 목 모터의
     present_position 을 sensor_msgs/JointState 로 발행. joint 이름이 URDF 와 정확히 일치해야 함)
  - 테스트만 할 거라면 joint_state_publisher_gui 로 대체 가능 (use_jsp_gui:=true)

사용:
  # 기본 (패키지 안 URDF 자동 사용)
  ros2 launch humanoid_perception robot_tf_bringup.launch.py

  # joint_states 가 아직 없을 때 (TF 시각 점검용 - 슬라이더로 관절 조작)
  ros2 launch humanoid_perception robot_tf_bringup.launch.py use_jsp_gui:=true

  # 카메라 없이 로봇 TF tree 만
  ros2 launch humanoid_perception robot_tf_bringup.launch.py enable_camera:=false

  # 다른 URDF 사용
  ros2 launch humanoid_perception robot_tf_bringup.launch.py urdf:=/path/to/other.urdf

확인:
  ros2 run tf2_tools view_frames
  ros2 run tf2_ros tf2_echo base_link camera_color_optical_frame
"""
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, OpaqueFunction,
)
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def _setup(context, *args, **kwargs):
    urdf_path   = LaunchConfiguration('urdf').perform(context)
    use_jsp_gui = LaunchConfiguration('use_jsp_gui').perform(context).lower() == 'true'
    enable_cam  = LaunchConfiguration('enable_camera').perform(context).lower() == 'true'
    rs_profile  = LaunchConfiguration('rs_camera_profile').perform(context)

    if not os.path.isfile(urdf_path):
        raise RuntimeError(
            f"URDF 파일을 찾을 수 없습니다: {urdf_path}\n"
            f"  - colcon build 를 했는지 확인 (urdf/ 가 share 로 install 되어야 함)\n"
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

    return actions


def generate_launch_description():
    # 패키지 share 안의 기본 URDF 경로를 자동 계산.
    # setup.py 의 data_files 에서 urdf/*.urdf 를 share/<pkg>/urdf/ 로 install 해야 동작.
    default_urdf = os.path.join(
        get_package_share_directory('humanoid_perception'),
        'urdf', 'XM_robot03_output.urdf')

    return LaunchDescription([
        DeclareLaunchArgument('urdf', default_value=default_urdf,
            description='펼쳐진 .urdf 파일 경로. 기본값은 패키지 share 안의 URDF'),
        DeclareLaunchArgument('use_jsp_gui', default_value='false',
            description='true 면 joint_state_publisher_gui 로 관절 수동 조작 (TF 점검용)'),
        DeclareLaunchArgument('enable_camera', default_value='true',
            description='RealSense D405 실행 여부'),
        DeclareLaunchArgument('rs_camera_profile', default_value='848x480x30',
            description='RealSense color+depth 프로파일 (WxHxFPS)'),

        OpaqueFunction(function=_setup),
    ])
