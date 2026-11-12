#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================================================
# game_stack.launch.py  (humanoid_bringup)  — [기능 그룹 ⑤: 게임/전략]
# --------------------------------------------------------------------------
# "두뇌"에 해당하는 의사결정·협응 레이어를 한 번에 켠다.
# 기존 leaf 런치를 include 만 하며, robot/team/player 자동결정 규칙은
# 각 leaf 가 그대로 처리하므로 여기서는 인자만 통과시킨다.
#
#   포함 런치
#   ----------------------------------------------------------------------
#   1) game_controller.launch.py  (humanoid_game_controller)
#        · game_controller_receiver : UDP 3838 → /game_state
#        · game_state_to_lock        : /game_state → game_stop_lock(twist_mux 정지락)
#        · game_controller_return    : 개인상태 → GC(:3939) 송신 (enable_return)
#   2) strategy.launch.py         (humanoid_strategy)
#        · role_assigner_node : /game_state + /robot_state → /strategy/role
#        · strategy_fsm_node  : 역할/단계 → /pose_goal 포지셔닝 + 스킬 enable 주인
#   3) team_comm.launch.py        (humanoid_strategy)        use_team_comm
#        · team_comm_node : 자세/공/넘어짐 + 역할/단계/공거리를 UDP
#          broadcast(포트 10000+team_number)로 공유 → /team_comm/state(거리중재 owner)
#
# [team_number / player_number 자동 결정]  ← 세 leaf 모두 동일 규칙
#   · robot 인자가 비면 hostname 앞부분 (robo04-desktop → robo04)
#   · humanoid_bringup/config/robot_params.yaml 에서
#       team_number(common) / player_number(<robot>) 읽음
#   우선순위: launch 인자 > robot_params.yaml > 기본값(team 35 / player 1)
#   → 각 로봇에서 그냥 띄우면 그 로봇 번호로 자동 적용된다.
#
# [전제]
#   · 실제 정지 효과(game_stop_lock)는 control.launch.py 의 twist_mux 가
#     이미 game_stop_lock 을 구독하고 있어야 발생(humanoid_bringup 에 설정됨).
#   · strategy_fsm 이 실제로 걸어가려면 로컬라이제이션 TF + pose_goal/ball_follow/
#     vision_servo(behavior_stack)가 함께 떠 있어야 한다.
#
# [통신 예산 주의]  HSL 2026: 게임당 UDP 12000개. team_comm send_rate_hz 로 조정.
#
# [실행 예]
#   ros2 launch humanoid_bringup game_stack.launch.py                 # 전부 자동
#   ros2 launch humanoid_bringup game_stack.launch.py robot:=robo03
#   ros2 launch humanoid_bringup game_stack.launch.py use_team_comm:=false
#   ros2 launch humanoid_bringup game_stack.launch.py mirror_field:=true
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
    # ── 공유 식별 인자(빈 값이면 각 leaf 가 자동 결정) ──────────
    robot = LaunchConfiguration("robot")
    team_number = LaunchConfiguration("team_number")
    player_number = LaunchConfiguration("player_number")

    # ── 토글/세부 인자 ──────────────────────────────────────
    use_team_comm = LaunchConfiguration("use_team_comm")
    enable_return = LaunchConfiguration("enable_return")
    mirror_field = LaunchConfiguration("mirror_field")
    gc_address = LaunchConfiguration("gc_address")
    send_rate_hz = LaunchConfiguration("send_rate_hz")
    broadcast_address = LaunchConfiguration("broadcast_address")

    args = [
        DeclareLaunchArgument(
            "robot", default_value="",
            description="robot_params.yaml 섹션 선택. 비면 hostname 앞부분(robo04 등)"),
        DeclareLaunchArgument(
            "team_number", default_value="",
            description="우리 팀 번호. 비면 robot_params.yaml common(기본 35)"),
        DeclareLaunchArgument(
            "player_number", default_value="",
            description="이 로봇 선수 번호. 비면 robot_params.yaml 에서 자동"),
        DeclareLaunchArgument(
            "use_team_comm", default_value="true",
            description="팀 통신(team_comm) 포함 여부. 단독 테스트 시 false 로 끔"),
        DeclareLaunchArgument(
            "enable_return", default_value="true",
            description="개인 상태 GC 송신(game_controller_return) on/off"),
        DeclareLaunchArgument(
            "mirror_field", default_value="false",
            description="진영(코트) 반대일 때 pose 를 점대칭으로 송신"),
        DeclareLaunchArgument(
            "gc_address", default_value="",
            description="GameController IP. 비면 3838 브로드캐스트로 자동 학습"),
        DeclareLaunchArgument(
            "send_rate_hz", default_value="2.0",
            description="team_comm 송신 주기[Hz]. 게임당 12000개 예산 내에서 조정"),
        DeclareLaunchArgument(
            "broadcast_address", default_value="255.255.255.255",
            description="team_comm UDP broadcast 대상 주소"),
    ]

    # 1) GameController 브링업(수신/정지락/송신)
    game_controller = _include(
        "humanoid_game_controller", "game_controller.launch.py",
        launch_arguments={
            "robot": robot,
            "team_number": team_number,
            "player_number": player_number,
            "enable_return": enable_return,
            "gc_address": gc_address,
            "mirror_field": mirror_field,
        },
    )

    # 2) 전략(역할 배정 + FSM)
    strategy = _include(
        "humanoid_strategy", "strategy.launch.py",
        launch_arguments={
            "robot": robot,
            "player_number": player_number,
        },
    )

    # 3) 팀 통신 — use_team_comm:=true 일 때만
    team_comm = _include(
        "humanoid_strategy", "team_comm.launch.py",
        condition=IfCondition(use_team_comm),
        launch_arguments={
            "robot": robot,
            "team_number": team_number,
            "player_number": player_number,
            "send_rate_hz": send_rate_hz,
            "broadcast_address": broadcast_address,
        },
    )

    return LaunchDescription(args + [game_controller, strategy, team_comm])
