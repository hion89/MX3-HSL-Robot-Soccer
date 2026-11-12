#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================================================
# perception_stack.launch.py  (humanoid_bringup)  — [기능 그룹 ②: 인지]
# --------------------------------------------------------------------------
# "눈"에 해당하는 인지 레이어를 한 번에 켠다. 기존 leaf 런치를 include 만 하며
# (단일 소스 유지 — 노드 파라미터 원본은 각 패키지 런치/yaml 이 그대로 소유),
# 그 위에 자주 쓰는 토글/인자만 위로 노출한다.
#
#   포함 런치(humanoid_perception)
#   ----------------------------------------------------------------------
#   1) perception_bringup.launch.py   (항상 포함 — 이 그룹의 핵심)
#        · robot_tf_bringup(enable_camera:=false) → robot_state_publisher
#          (base_link↔camera 등 로봇 TF 발행. 카메라 장치는 안 잡음)
#        · perception_3d_clean.launch.py → RealSense(D405) + YOLO 3D 검출
#          (/perception/detections_3d)
#        · ball_map.launch.py → 공 검출을 map 좌표로 투영(/ball/map_point, 마커)
#   2) depth_obstacle.launch.py       (use_obstacle:=true 일 때만)
#        · 단거리 깊이 장애물 검출 → /obstacle/avoid_hint
#          (pose_goal_controller 가 회피에 소비)
#
# [중복 없음 — 합치면서 점검한 사항]
#   · robot_state_publisher 는 여기(robot_tf_bringup) 한 곳에서만 발행한다.
#     하드웨어(dynamixel_system)는 /joint_states 만, ekf/vision_localization 은
#     robot_state_publisher 를 안 띄운다. → TF 충돌 없음.
#   · RealSense 카메라(rs_launch)도 perception_3d_clean 한 곳에서만 잡는다.
#
# [전제]
#   · map→base_footprint 앞단(EKF)은 별도(ekf_localization). 그게 없으면 공 map
#     투영 노드는 TF 조회 실패로 '조용히 대기'(에러 아님 — 로컬라이제이션 미가동 의미).
#
# [실행 예]
#   ros2 launch humanoid_bringup perception_stack.launch.py
#   # 장애물 검출 빼고 싶을 때:
#   ros2 launch humanoid_bringup perception_stack.launch.py use_obstacle:=false
#   # 카메라 프로파일/신뢰도 바꾸기:
#   ros2 launch humanoid_bringup perception_stack.launch.py \
#       rs_camera_profile:=640x480x30 conf:=0.30
# ==========================================================================

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _include(pkg: str, launch_file: str, condition=None, launch_arguments=None):
    """지정 패키지 share/launch 에서 launch 파일을 찾아 include 하는 헬퍼.

    launch_arguments: {str: str|Substitution} 딕셔너리(없으면 빈 dict).
    condition: IfCondition 등(없으면 항상 포함).
    """
    path = os.path.join(
        get_package_share_directory(pkg), "launch", launch_file)
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(path),
        condition=condition,
        launch_arguments=(launch_arguments or {}).items(),
    )


def generate_launch_description():
    # ── 위로 노출하는 토글/인자 ──────────────────────────────
    use_obstacle = LaunchConfiguration("use_obstacle")
    rs_camera_profile = LaunchConfiguration("rs_camera_profile")
    conf = LaunchConfiguration("conf")
    use_ball_map = LaunchConfiguration("use_ball_map")
    ball_marker_diameter_m = LaunchConfiguration("ball_marker_diameter_m")
    obstacle_max_range = LaunchConfiguration("obstacle_max_range")

    args = [
        DeclareLaunchArgument(
            "use_obstacle", default_value="true",
            description="단거리 깊이 장애물 검출(depth_obstacle) 포함 여부. "
                        "현재 YOLO/메모리 경로가 1차라 필요 없으면 false 로 끈다"),
        DeclareLaunchArgument(
            "rs_camera_profile", default_value="848x480x30",
            description="RealSense color/depth 프로파일 (WxHxFPS)"),
        DeclareLaunchArgument(
            "conf", default_value="0.25",
            description="YOLO confidence 임계값"),
        DeclareLaunchArgument(
            "use_ball_map", default_value="true",
            description="공 → map 투영 노드(ball_map_projector) 실행 여부"),
        DeclareLaunchArgument(
            "ball_marker_diameter_m", default_value="0.0",
            description="공 RViz 마커 지름[m]. 0=자동(2×반지름), >0=마커만 강제로 키움"),
        DeclareLaunchArgument(
            "obstacle_max_range", default_value="0.5",
            description="depth 장애물 전방 처리 최대거리[m] (단거리 범퍼 기본 0.5)"),
    ]

    # 1) 인지 핵심 — 항상 포함(robot TF + 카메라 + YOLO3D + 공 map 투영)
    perception_bringup = _include(
        "humanoid_perception", "perception_bringup.launch.py",
        launch_arguments={
            "rs_camera_profile": rs_camera_profile,
            "conf": conf,
            "use_ball_map": use_ball_map,
            "ball_marker_diameter_m": ball_marker_diameter_m,
        },
    )

    # 2) 단거리 깊이 장애물 검출 — use_obstacle:=true 일 때만
    depth_obstacle = _include(
        "humanoid_perception", "depth_obstacle.launch.py",
        condition=IfCondition(use_obstacle),
        launch_arguments={
            "max_range": obstacle_max_range,
        },
    )

    return LaunchDescription(args + [perception_bringup, depth_obstacle])
