#!/usr/bin/env python3
# ==========================================================================
# game_controller.launch.py
# --------------------------------------------------------------------------
# 1단계 GameController 브링업:
#   1) game_controller_receiver : UDP 3838 수신 → /game_state
#   2) game_state_to_lock        : /game_state → game_stop_lock (twist_mux 정지락)
#   3) game_controller_return    : 개인상태(자세/공/넘어짐) → GC(:3939) 송신
#                                  enable_return:=false 로 끌 수 있음(기본 켜짐)
#
# [team_number / player_number 자동 결정]  ← 핵심
#   control.launch.py 와 동일한 방식으로 robot 을 해석한다:
#     - robot 인자가 비면 hostname 앞부분 사용 (robo04-desktop → robo04)
#     - humanoid_bringup/config/robot_params.yaml 에서 값을 읽음
#         · team_number   : common 섹션(팀 전체 공통)
#         · player_number : <robot> 섹션(로봇별)
#   우선순위: launch 인자(명시) > robot_params.yaml > 기본값(team 35 / player 1)
#   즉 robo04 에서 그냥 띄우면 team_number=35, player_number=4 가 자동 적용된다.
#
# [실행 예]
#   ros2 launch humanoid_game_controller game_controller.launch.py            # 전부 자동
#   ros2 launch humanoid_game_controller game_controller.launch.py robot:=robo03
#   ros2 launch humanoid_game_controller game_controller.launch.py player_number:=2   # 수동 override
#
#   - 이 launch 는 game_stop_lock 만 낸다. 실제 정지 효과는 control.launch.py 의
#     twist_mux 가 game_stop_lock 을 구독하고 있어야 발생한다(이미 설정돼 있음).
# ==========================================================================
import os
import socket
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.logging import get_logger
from launch_ros.actions import Node


def _resolve_robot_name(name: str) -> str:
    """robot 인자가 비면 hostname 앞부분으로 (robo04-desktop → robo04).
    control.launch.py 의 _resolve_robot_name 과 동일한 규칙을 쓴다."""
    name = (name or "").strip()
    if name:
        return name
    host = socket.gethostname()
    return host.split("-")[0].split(".")[0]


def _load_merged_params(robot: str):
    """robot_params.yaml 에서 common + <robot> 병합본을 반환(robot 우선).
    control.launch.py 의 _load_robot_params 와 동일한 병합 규칙."""
    cfg = os.path.join(
        get_package_share_directory("humanoid_bringup"),
        "config", "robot_params.yaml")
    data = {}
    if os.path.isfile(cfg):
        with open(cfg, "r") as f:
            data = yaml.safe_load(f) or {}
    merged = dict(data.get("common", {}) or {})
    merged.update(data.get(robot, {}) or {})
    return merged, cfg


def _resolve_int(arg_val: str, merged: dict, key: str, default: int):
    """우선순위: launch 인자(명시) > robot_params.yaml > 기본값.
    반환: (값:int, 출처설명:str)."""
    arg_val = (arg_val or "").strip()
    if arg_val:
        return int(arg_val), f"launch 인자({key}:=)"
    if key in merged:
        return int(merged[key]), "robot_params.yaml"
    return default, f"기본값({default}) — yaml 에 {key} 없음"


def _setup(context, *args, **kwargs):
    log = get_logger("game_controller.launch")

    robot = _resolve_robot_name(LaunchConfiguration("robot").perform(context))
    merged, cfg = _load_merged_params(robot)

    # team_number: 인자 > robot_params.yaml common > 기본 35
    team_number, team_src = _resolve_int(
        LaunchConfiguration("team_number").perform(context), merged, "team_number", 35)
    # player_number: 인자 > robot_params.yaml <robot> > 기본 1
    player_number, pn_src = _resolve_int(
        LaunchConfiguration("player_number").perform(context), merged, "player_number", 1)

    log.info(
        f"[game_controller] robot='{robot}' "
        f"team_number={team_number}({team_src}) "
        f"player_number={player_number}({pn_src})  [{cfg}]")

    return [
        # ── 1) GameController 수신 노드 ──
        Node(
            package="humanoid_game_controller",
            executable="game_controller_receiver",
            name="game_controller_receiver",
            output="screen",
            parameters=[{
                "team_number": team_number,
                "listen_port": 3838,
                "bind_address": "0.0.0.0",
                "log_throttle_sec": 2.0,
            }],
        ),
        # ── 2) game_stop_lock 매핑 노드 (자동 결정된 player_number 주입) ──
        Node(
            package="humanoid_game_controller",
            executable="game_state_to_lock",
            name="game_state_to_lock",
            output="screen",
            parameters=[{
                "player_number": player_number,
                "stale_timeout_sec": 2.0,
                "publish_rate_hz": 10.0,
            }],
        ),
        # ── 3) 개인 상태 송신 노드 (자세/공/넘어짐 → GC:3939) ──
        #   enable_return:=false 면 생성하지 않음. gc_address 비면 3838 로 자동 학습.
        Node(
            package="humanoid_game_controller",
            executable="game_controller_return",
            name="game_controller_return",
            output="screen",
            condition=IfCondition(LaunchConfiguration("enable_return")),
            parameters=[{
                "team_number": team_number,
                "player_number": player_number,
                "gc_address": LaunchConfiguration("gc_address"),
                "return_port": 3939,
                "send_rate_hz": 2.0,
                "mirror_field": LaunchConfiguration("mirror_field"),
            }],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "team_number", default_value="",
            description="우리 팀 번호. 비면 robot_params.yaml common 에서 자동(기본 35)"),
        DeclareLaunchArgument(
            "robot", default_value="",
            description="robot_params.yaml 섹션 선택. 비면 hostname 앞부분(robo04 등)"),
        DeclareLaunchArgument(
            "player_number", default_value="",
            description="이 로봇 선수 번호. 비면 robot_params.yaml 에서 자동 결정"),
        DeclareLaunchArgument(
            "enable_return", default_value="true",
            description="개인 상태 송신(game_controller_return) 노드 on/off"),
        DeclareLaunchArgument(
            "gc_address", default_value="",
            description="GameController IP. 비면 3838 브로드캐스트로 자동 학습"),
        DeclareLaunchArgument(
            "mirror_field", default_value="false",
            description="진영(코트) 반대일 때 pose 를 점대칭으로 송신"),
        OpaqueFunction(function=_setup),
    ])
