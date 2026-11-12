#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================================================
# behavior_stack.launch.py  (humanoid_bringup)  — [기능 그룹 ④: 행동/스킬]
# --------------------------------------------------------------------------
# 전략(strategy)이 enable/disable 로 부려 쓰는 "스킬" 노드들을 한 번에 켠다.
# 기존 leaf 런치를 include 만 하며, 각 스킬을 개별 토글할 수 있게 노출한다.
#
#   포함 런치
#   ----------------------------------------------------------------------
#   1) pose_goal.launch.py        (humanoid_localization)  use_pose_goal
#        · pose_goal_controller_node: /pose_goal(PoseStamped) → cmd_vel_nav 주행
#          (+ /obstacle/avoid_hint 회피 합성). 모든 위치이동의 공통 1차 컨트롤러.
#   2) ball_follow.launch.py      (humanoid_localization)  use_ball_follow
#        · ball_follow_nav_node: 공 근처까지 rough approach (/pose_goal 발행)
#   3) vision_servo.launch.py     (humanoid_localization)  use_vision_servo
#        · visual_servo_kick_node: 공-골대 축 최종 정렬 + 킥(cmd_vel_behavior)
#   4) neck_ball_tracker.launch.py(humanoid_serial_bridge) use_neck_tracker
#        · neck_ball_tracker_node: '시선' 중재자.
#          1·2·3 이 보내는 /neck/mode_request/* (String, 예: BALL_TRACKING /
#          RICH_MARKER / SWEEP_DETAILED)를 받아 목을 그 모드로 구동한다.
#          → 시선 요청의 '소비자'라서 이 그룹에 함께 둔다.
#
# [중요 — 경기 중 enable 주인은 strategy]
#   pose_goal / ball_follow / vision_servo 는 leaf 기본 auto_start:=false 라
#   여기서는 '대기(IDLE)' 상태로만 뜬다. 실제 enable 토글은 strategy_fsm 이
#   소유한다(이 그룹은 스킬을 '깔아두는' 역할). 단독 테스트가 필요하면 각 leaf
#   런치를 auto_start:=true 로 직접 띄워라(여기서는 일부러 노출하지 않음).
#
# [전제 — 이 그룹이 의미 있으려면 아래가 먼저 떠 있어야 함]
#   · 보행 정책 actor 가 /cmd_vel 추종 중(humanoid_bringup control.launch.py)
#   · twist_mux 슬롯(cmd_vel_nav / cmd_vel_behavior) 활성
#   · 로컬라이제이션 TF(map→base_footprint): ekf_localization 런치
#   · 인지: /perception/detections_3d, /ball/map_point (perception_stack)
#
# [실행 예]
#   ros2 launch humanoid_bringup behavior_stack.launch.py
#   # 킥 정렬(vision_servo)만 빼고:
#   ros2 launch humanoid_bringup behavior_stack.launch.py use_vision_servo:=false
#   # 파랑 진영 공격(공격 골대 x 부호) + 회피 끔:
#   ros2 launch humanoid_bringup behavior_stack.launch.py \
#       target_goal_x:=-1.95 avoid_enable:=false
# ==========================================================================

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _include(pkg: str, launch_file: str, condition=None, launch_arguments=None):
    """지정 패키지 share/launch 에서 launch 파일을 찾아 include 하는 헬퍼."""
    path = os.path.join(
        get_package_share_directory(pkg), "launch", launch_file)
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(path),
        condition=condition,
        launch_arguments=(launch_arguments or {}).items(),
    )


def generate_launch_description():
    # ── 스킬별 on/off 토글 ──────────────────────────────────
    use_pose_goal = LaunchConfiguration("use_pose_goal")
    use_ball_follow = LaunchConfiguration("use_ball_follow")
    use_vision_servo = LaunchConfiguration("use_vision_servo")
    use_neck_tracker = LaunchConfiguration("use_neck_tracker")

    # ── 자주 바꾸는 pass-through 인자 ──────────────────────────
    avoid_enable = LaunchConfiguration("avoid_enable")       # pose_goal 회피 합성
    goal_source = LaunchConfiguration("goal_source")         # vision_servo 골대 방위 소스
    target_goal_x = LaunchConfiguration("target_goal_x")     # vision_servo 공격 골대 x[m]
    target_goal_y = LaunchConfiguration("target_goal_y")     # vision_servo 공격 골대 y[m]

    args = [
        DeclareLaunchArgument(
            "use_pose_goal", default_value="true",
            description="pose_goal_controller(공통 위치이동 컨트롤러) 포함 여부"),
        DeclareLaunchArgument(
            "use_ball_follow", default_value="true",
            description="ball_follow_nav(공 rough approach) 포함 여부"),
        DeclareLaunchArgument(
            "use_vision_servo", default_value="true",
            description="visual_servo_kick(최종 정렬+킥) 포함 여부"),
        DeclareLaunchArgument(
            "use_neck_tracker", default_value="true",
            description="neck_ball_tracker(시선 중재자) 포함 여부"),
        DeclareLaunchArgument(
            "avoid_enable", default_value="true",
            description="pose_goal 의 /obstacle/avoid_hint 회피 합성 on/off"),
        DeclareLaunchArgument(
            "goal_source", default_value="auto",
            description="vision_servo 골대 방위 소스: 'auto'/'vision'/'localization'"),
        DeclareLaunchArgument(
            "target_goal_x", default_value="1.95",
            description="vision_servo localization 폴백 공격골대 x[m]. "
                        "노랑 진영 +1.95 / 파랑 진영 -1.95"),
        DeclareLaunchArgument(
            "target_goal_y", default_value="0.0",
            description="vision_servo localization 폴백 공격골대 y[m]"),
    ]

    # 1) 공통 위치이동 컨트롤러 (auto_start 는 leaf 기본 false 유지 — strategy 가 enable)
    pose_goal = _include(
        "humanoid_localization", "pose_goal.launch.py",
        condition=IfCondition(use_pose_goal),
        launch_arguments={
            "avoid_enable": avoid_enable,
        },
    )

    # 2) 공 rough approach
    ball_follow = _include(
        "humanoid_localization", "ball_follow.launch.py",
        condition=IfCondition(use_ball_follow),
    )

    # 3) 최종 정렬 + 킥
    vision_servo = _include(
        "humanoid_localization", "vision_servo.launch.py",
        condition=IfCondition(use_vision_servo),
        launch_arguments={
            "goal_source": goal_source,
            "target_goal_x": target_goal_x,
            "target_goal_y": target_goal_y,
        },
    )

    # 4) 시선 중재자 (mode_request 소비자) — leaf 기본 yaml 그대로 사용
    neck_tracker = _include(
        "humanoid_serial_bridge", "neck_ball_tracker.launch.py",
        condition=IfCondition(use_neck_tracker),
    )

    return LaunchDescription(
        args + [pose_goal, ball_follow, vision_servo, neck_tracker])
