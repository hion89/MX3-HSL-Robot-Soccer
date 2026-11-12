# -*- coding: utf-8 -*-
# =============================================================================
#  ekf_localization.launch.py   (Phase 3)
#
#  비전 측정 + 보행 오도메트리를 robot_localization(EKF)으로 융합한다.
#  명령 하나로 전체 로컬라이제이션 스택을 띄운다(TF 충돌 방지 포함).
#
#  [이 런치가 띄우는 것]
#   1. vision_localization.launch.py 포함
#        - field_landmark_localizer_node (/localization/vision_pose + 디버그 마커)
#        - field_lines_node (경기장 라인)
#        - ★ publish_map_odom_identity:=false 로 강제 → Phase 2 의 static map→odom 을 끈다
#          (이제 map→odom 의 주인은 EKF. 둘 다 켜면 TF 가 싸운다)
#   2. walk_odometry_node          : /odom + odom→base_link TF (cmd_vel + IMU yaw)
#   3. ekf_filter_node             : /odom(속도) + /vision_pose(절대) 융합 → map→odom TF,
#                                    /odometry/filtered 발행
#
#  [함께 따로 켜야 하는 것 — 검출/TF/센서 소스]
#    ros2 launch humanoid_perception robot_tf_bringup.launch.py enable_camera:=false  # 로봇 TF
#    ros2 launch humanoid_perception perception_3d_clean.launch.py                    # D405 + YOLO 3D
#    (+ /joint_states 와 /imu/data 가 나오도록 평소 bringup. walk_odometry 는 /imu/data 필요)
#
#  [전제] robot_localization 설치 필요:
#    sudo apt install ros-humble-robot-localization
#
#  [검증(증분)]
#   3a) /odom 확인:   ros2 topic echo /odom --once  /  ros2 run tf2_ros tf2_echo odom base_link
#   3b) EKF 확인:     ros2 topic echo /odometry/filtered --once
#                     ros2 run tf2_ros tf2_echo map odom   (EKF 가 발행해야 함)
#   3c) RViz: VisionPose(주황, 떨림) 옆에 EKF Filtered(초록, 부드러움)가 같이 뜬다.
#       랜드마크 가렸다 풀어도 EKF 가 끊김 없이 이어가는지 본다.
# =============================================================================
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('humanoid_localization')
    odom_params = os.path.join(pkg_share, 'config', 'odom_calib.yaml')
    ekf_params = os.path.join(pkg_share, 'config', 'ekf.yaml')
    vision_launch = os.path.join(pkg_share, 'launch', 'vision_localization.launch.py')

    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='RViz 함께 실행(Phase 3 는 기본 켬)')

    # ★ 경기장 프로파일 — 포함하는 vision_localization 으로 그대로 전달한다.
    #   이 인자가 없으면 vision 런치가 자기 기본값(practice=3.9x2.6)으로 떠서,
    #   대회장(9x6)에서 ekf 런치만 켜면 소형 필드로 측위하는 사고가 난다.
    #     practice  = 연습장 3.9 x 2.6
    #     s_field   = 대회 9 x 6 (HSL 2026 small division; "competition" 별칭도 동일)
    #   → ekf 런치 한 줄로 localizer + 재진입 시드 + 라인이 전부 같이 전환된다.
    field_profile_arg = DeclareLaunchArgument(
        'field_profile', default_value='practice',
        description='경기장 규격: practice(연습장 3.9x2.6) / s_field(대회 9x6). '
                    '대회 때는 field_profile:=s_field 한 줄로 전환.')

    # ---- 1) 비전 런치 포함 (static map→odom 끄고, EKF 가 그 역할을 맡음) ----
    vision = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(vision_launch),
        launch_arguments={
            'publish_map_odom_identity': 'false',  # ★ EKF 가 map→odom 발행하므로 static 끔
            'start_field_lines': 'true',
            'rviz': LaunchConfiguration('rviz'),
            # ★ 경기장 프로파일을 vision 으로 전달(미전달 시 practice 로 떠 버림)
            'field_profile': LaunchConfiguration('field_profile'),
        }.items(),
    )

    # ---- 2) 보행 오도메트리 (/odom + odom→base_link TF) ----
    walk_odom = Node(
        package='humanoid_localization',
        executable='walk_odometry_node',
        name='walk_odometry_node',
        output='screen',
        parameters=[odom_params],
        # ★ /initialpose 는 EKF(map 프레임)만 리셋해야 한다.
        #   odom 소스인 walk_odometry 가 같이 리셋하면 odom→base_link 가 튀어
        #   EKF 의 map→odom 과 싸운다(화면에서 TF·화살표가 진동하며 되돌아감).
        #   → walk_odometry 의 /initialpose 구독을 죽은 토픽으로 remap 해 비활성화.
        remappings=[('/initialpose', '/initialpose_walk_odom_disabled')],
    )

    # ---- 3) robot_localization EKF (/odom + /vision_pose 융합 → map→odom TF) ----
    ekf = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[ekf_params],
        # ★ RViz "2D Pose Estimate"(/initialpose)로 EKF 상태를 리셋하게 한다.
        #   robot_localization 은 기본적으로 자기 'set_pose' 토픽만 듣는다.
        #   이걸 /initialpose 로 remap 하면, 클릭 한 번에 EKF 가 그 위치로 점프하고
        #   거기서부터 vision 이 거부 없이 추적을 시작한다(초기화/재위치).
        remappings=[('set_pose', '/initialpose')],
    )

    return LaunchDescription([rviz_arg, field_profile_arg, vision, walk_odom, ekf])