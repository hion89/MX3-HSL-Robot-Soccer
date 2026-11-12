# -*- coding: utf-8 -*-
# =============================================================================
#  vision_localization.launch.py
#
#  비전 로컬라이저(field_landmark_localizer_node) + RViz 검토 환경을 띄운다.
#
#  [기본 동작 — Phase 2 standalone 검증용]
#  --------------------------------------------------------------------------
#  기본값으로 다음을 함께 띄워, 실제 카메라 입력을 RViz 로 바로 검증할 수 있게 한다:
#    1) field_landmark_localizer_node : 검출→/localization/vision_pose + 디버그 마커
#    2) map→odom 항등(static) TF       : EKF 가 아직 없는 Phase 2 에서 'map' 프레임을
#                                        TF 트리에 붙여 RViz 가 map 기준으로 표시 가능하게.
#                                        ※ Phase 3 에서 EKF 가 map→odom 을 발행하면
#                                          publish_map_odom_identity:=false 로 끈다.
#    3) field_lines_node               : 경기장 라인(/field_markers, map) 시각화
#
#  RViz 까지 한 번에:  rviz:=true
#
#  [함께 따로 켜야 하는 것 — 실제 검출/TF 소스]
#    ros2 launch humanoid_perception robot_tf_bringup.launch.py        # base_link↔camera TF + D405
#    ros2 launch humanoid_perception perception_3d_clean.launch.py     # YOLO 3D → /perception/detections_3d
#
#  [검증 절차]
#    - 로봇을 경기장 안 "알려진 위치"에 둔다(가만히).
#    - 그 위치를 config 의 initial_x/y/yaw 로 맞추거나 RViz "2D Pose Estimate"(/initialpose)로 지정.
#    - RViz 에서 확인:
#        · landmarks(클래스별 구)   = 솔버가 아는 랜드마크 지도
#        · matched(청록 선)         = 지금 로봇이 보고 매칭한 랜드마크
#        · observed(자홍 구)        = 측정이 둔 랜드마크 위치(실제 구와 겹치면 잘 맞는 것)
#        · robot(주황 화살표)       = 추정한 내 위치/방향
#      → 화살표가 실제 둔 위치와 맞고, observed 구가 landmarks 구에 겹치면 측정 정상.
#    ※ 로봇 모델(odom 기준)과 추정 화살표(map 기준)가 어긋나 보이는 건 정상 —
#      그 둘을 정렬하는 게 Phase 3(EKF).
# =============================================================================
import os
import socket

import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def _resolve_reentry_x_inset() -> float:
    """reentry_positions.yaml(이 패키지 config) 의 <robot>.x_inset_m 를 읽는다.

    로봇 선택은 hostname 앞부분(control.launch 와 동일 규약: robo04-desktop → robo04).
    파일없음/키없음/파싱오류 등 "어떤" 실패에도 0.0 으로 폴백한다.

    이 값은 relocalize_button_node 의 x_inset_m(시드 x 오프셋)으로 들어간다.
    재진입 시드 위치는 "로컬라이제이션 관심사"이므로 이 패키지가 직접 소유한다
    (하드웨어 quirk 인 bringup/robot_params.yaml 과 성격이 달라 분리).
    → 이 런치는 자기 패키지 설정만 읽으므로 단독 실행 시 외부 의존이 없다.
    """
    try:
        robot = socket.gethostname().split('-')[0].split('.')[0]
        cfg = os.path.join(
            get_package_share_directory('humanoid_localization'),
            'config', 'reentry_positions.yaml')
        with open(cfg, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        robocfg = data.get(robot, {}) or {}
        return float(robocfg.get('x_inset_m', 0.0))
    except Exception:
        # 파일없음/파싱오류 → 폴백(시드 오프셋 0)
        return 0.0


def generate_launch_description():
    pkg_share = get_package_share_directory('humanoid_localization')
    default_params = os.path.join(pkg_share, 'config', 'vision_localization.yaml')
    field_params = os.path.join(pkg_share, 'config', 'field.yaml')
    default_rviz = os.path.join(pkg_share, 'rviz', 'vision_localization.rviz')

    # ---- 런치 인자 ----
    params_arg = DeclareLaunchArgument(
        'params_file', default_value=default_params,
        description='field_landmark_localizer_node 파라미터 yaml')
    map_odom_arg = DeclareLaunchArgument(
        'publish_map_odom_identity', default_value='true',
        description='map→odom 항등 TF 발행(EKF 없는 Phase 2 검증용). Phase 3 에선 false')
    field_arg = DeclareLaunchArgument(
        'start_field_lines', default_value='true',
        description='경기장 라인 시각화(field_lines_node) 함께 실행')
    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='false',
        description='RViz 도 함께 실행')
    rviz_cfg_arg = DeclareLaunchArgument(
        'rviz_config', default_value=default_rviz,
        description='RViz 설정 파일 경로')
    profile_arg = DeclareLaunchArgument(
        'field_profile', default_value='practice',
        description="경기장 규격: practice(연습장 3.9x2.6) / competition(대회 9x6). "
                    "localizer 와 relocalize 노드에 '같은 값'으로 적용된다. "
                    "대회 때는 field_profile:=competition 한 줄로 전환.")
    relocalize_arg = DeclareLaunchArgument(
        'start_relocalize_button', default_value='true',
        description='SW1+실행버튼 재진입 시드 노드(relocalize_button_node) 실행')
    inset_arg = DeclareLaunchArgument(
        'reentry_x_inset_m', default_value=str(_resolve_reentry_x_inset()),
        description="재진입 시드 x 오프셋(m). 기본값은 이 패키지 config/reentry_positions.yaml "
                    "의 <hostname> 에서 자동 로드(로봇별로 다르게). "
                    "CLI 로 reentry_x_inset_m:=0.2 처럼 직접 덮어쓸 수도 있다.")

    # ---- 1) 비전 로컬라이저 노드 ----
    #   field_profile 은 launch 인자로 yaml 값을 덮어쓴다(last-wins).
    localizer = Node(
        package='humanoid_localization',
        executable='field_landmark_localizer_node',
        name='field_landmark_localizer_node',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {'field_profile': LaunchConfiguration('field_profile')},
        ],
    )

    # ---- 2) map→odom 항등 TF (Phase 2 에서 map 프레임을 트리에 붙임) ----
    #     인자: x y z yaw pitch roll parent child  → 전부 0 = 항등.
    static_map_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_map_odom_identity',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        condition=IfCondition(LaunchConfiguration('publish_map_odom_identity')),
    )

    # ---- 3) 경기장 라인 시각화 ----
    #   field_profile 을 localizer 와 '같은 값'으로 주입한다(last-wins).
    #   → field_profile:=s_field 한 줄로 측위 랜드마크 + 재진입 + RViz 라인이
    #     모두 같은 스케일로 동시에 전환된다(단일 출처).
    field_lines = Node(
        package='humanoid_localization',
        executable='field_lines_node',
        name='field_lines_node',
        output='screen',
        parameters=[
            field_params,
            {'field_profile': LaunchConfiguration('field_profile')},
        ],
        condition=IfCondition(LaunchConfiguration('start_field_lines')),
    )

    # ---- 4) RViz (옵션) ----
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    # ---- 5) 재진입 시드 노드 (SW1 좌/우 + 실행버튼 → /initialpose) ----
    #   field_profile 은 localizer 와 '같은 값'을 받아 시드 좌표를 같은 스케일로 맞춘다.
    #   세부 기하(outside_margin_m, x_inset_m, sw1_false_is_left 등)는 노드 기본값을
    #   쓰며, 필요 시 params_file 에 relocalize_button_node 섹션으로 덮어쓸 수 있다.
    relocalize_button = Node(
        package='humanoid_localization',
        executable='relocalize_button_node',
        name='relocalize_button_node',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {'field_profile': LaunchConfiguration('field_profile'),
             # 로봇별 시드 x 오프셋(robot_params.yaml reentry.<robot> → hostname 자동).
             # ParameterValue(value_type=float): 문자열 인자를 double 파라미터로 강제 변환.
             'x_inset_m': ParameterValue(
                 LaunchConfiguration('reentry_x_inset_m'), value_type=float)},
        ],
        condition=IfCondition(LaunchConfiguration('start_relocalize_button')),
    )

    return LaunchDescription([
        params_arg, map_odom_arg, field_arg, rviz_arg, rviz_cfg_arg,
        profile_arg, relocalize_arg, inset_arg,
        localizer, static_map_odom, field_lines, rviz, relocalize_button,
    ])
