#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================================================
# bringup.launch.py  (humanoid_bringup) — 최상위 진입점
# --------------------------------------------------------------------------
# KHUBER 전체 시스템을 레이어 단위로 켜고 끄며 한 번에 실행한다.
# 각 레이어는 인자로 토글하며, 기존 패키지의 launch 파일을 include 한다.
#
#   레이어         인자(기본값)        포함 내용
#   ----------------------------------------------------------------------
#   hardware    use_hardware(true)   serial_bridge/dynamixel_system.launch.py
#                                    (모터·IMU·neck·fall_manager 등 저수준)
#   control     use_control(true)    control.launch.py
#                                    (twist_mux + 보정 + actor 정책)
#   perception  use_perception(false) perception/perception.launch.py
#                                    (YOLO 공/마커 탐지) — 준비되면 true
#   teleop      use_teleop(false)    키보드 텔레옵(control.launch 로 전달)
#
# [사용 예]
#   # 전체(인지 제외):
#   ros2 launch humanoid_bringup bringup.launch.py
#
#   # 하드웨어 없이 정책 로직만(시뮬/디버그):
#   ros2 launch humanoid_bringup bringup.launch.py use_hardware:=false
#
#   # 하드웨어 + 텔레옵만(보행 수동 테스트):
#   ros2 launch humanoid_bringup bringup.launch.py use_control:=false use_teleop:=true
#
#   # 전부(인지 포함):
#   ros2 launch humanoid_bringup bringup.launch.py use_perception:=true
#
# [주의] include 하는 하위 launch 파일 이름이 본인 환경과 다르면
#        (예: perception 의 메인 launch 가 perception_clean.launch.py 라면)
#        아래 PERCEPTION_LAUNCH 상수만 바꾸면 된다.
# ==========================================================================

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

# 하위 launch 파일 이름 — 환경에 맞게 여기만 수정
HARDWARE_LAUNCH = "dynamixel_system.launch.py"   # humanoid_serial_bridge
CONTROL_LAUNCH = "control.launch.py"              # humanoid_bringup(본 패키지)
PERCEPTION_LAUNCH = "perception.launch.py"        # humanoid_perception


def _include(pkg: str, launch_file: str, condition=None, launch_arguments=None):
    """지정 패키지의 launch 파일을 share 경로에서 찾아 include 하는 헬퍼."""
    path = os.path.join(
        get_package_share_directory(pkg), "launch", launch_file)
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(path),
        condition=condition,
        launch_arguments=(launch_arguments or {}).items(),
    )


def generate_launch_description():
    use_hardware = LaunchConfiguration("use_hardware")
    use_control = LaunchConfiguration("use_control")
    use_perception = LaunchConfiguration("use_perception")
    use_teleop = LaunchConfiguration("use_teleop")
    use_kick = LaunchConfiguration("use_kick")
    # 킥 튜닝 인자 — control.launch.py 로 그대로 흘려보낸다.
    #   ★ 이걸 빠뜨리면 control.launch 의 기본값이 적용된다 ★
    #     실제로 kick_resume_hold 가 전달되지 않아, bringup 경로
    #     (= 비전 서보 테스트/match)에서만 킥 후 1.0초 정지가
    #     남아 있었다. 조이패드 경로는 control_joypad 가 직접
    #     넘겨서 문제가 없었고, 그래서 증상이 한쪽에만 나왔다.
    kick_resume_hold = LaunchConfiguration("kick_resume_hold")
    kick_ball_x = LaunchConfiguration("kick_ball_x")
    kick_cmd_vx = LaunchConfiguration("kick_cmd_vx")
    kick_duration = LaunchConfiguration("kick_duration")
    kick_start_hold = LaunchConfiguration("kick_start_hold")
    kick_strict_gate = LaunchConfiguration("kick_strict_gate")

    return LaunchDescription([
        DeclareLaunchArgument("use_hardware", default_value="true"),
        DeclareLaunchArgument("use_control", default_value="true"),
        DeclareLaunchArgument("use_perception", default_value="false"),
        DeclareLaunchArgument("use_teleop", default_value="false"),
        # 킥 러너(actor_onnx_walkkick_v1_runner) 실행 여부.
        # 기본 false(opt-in). match.launch.py 가 true 로 넘긴다.
        DeclareLaunchArgument("use_kick", default_value="false"),
        # 킥 튜닝 인자(기본값은 control.launch.py 와 동일하게 유지).
        DeclareLaunchArgument("kick_resume_hold", default_value="0.0"),
        DeclareLaunchArgument("kick_ball_x", default_value="0.30"),
        DeclareLaunchArgument("kick_cmd_vx", default_value="0.30"),
        DeclareLaunchArgument("kick_duration", default_value="2.0"),
        DeclareLaunchArgument("kick_start_hold", default_value="0.3"),
        DeclareLaunchArgument("kick_strict_gate", default_value="true"),
        # [2026-08 CPU 최적화] IMU 깨우기 절감 관련. 설명은 control.launch.py 참조.
        #   ★ 여기에 선언 + 아래 launch_arguments 에 전달, 둘 다 해야 먹는다.
        #     선언만 하고 전달을 빼먹으면 인자가 조용히 무시된다.
        # [2026-08] U2D2 SyncWrite→SyncRead 경합 제거(TX gap, 단위 us).
        #   근거는 u2d2_hardware.tx_gap() 주석 및 u2d2_txrx_bench.py 참조.
        #   실측: CycleTime 12.9ms → 약 4.4ms, 액션 지연은 불변.
        #   롤백: tx_gap_us:=0.0
        DeclareLaunchArgument("tx_gap_us", default_value="400.0"),
        DeclareLaunchArgument("gravity_source", default_value="derive"),
        DeclareLaunchArgument("fall_manager_imu_topic",
                              default_value="/imu/data_50hz"),

        # 1) 하드웨어 레이어
        _include("humanoid_serial_bridge", HARDWARE_LAUNCH,
                 condition=IfCondition(use_hardware),
                 # [2026-08] U2D2 반이중 경합 제거 인자를 하드웨어 레이어로 전달.
                 #   ★ 여기 전달을 빠뜨리면 dynamixel_system 의 기본값이 쓰이고
                 #     bringup 레벨 인자는 조용히 무시된다.
                 launch_arguments={"tx_gap_us": LaunchConfiguration("tx_gap_us")}),

        # 2) 제어 레이어 (텔레옵 토글을 그대로 control.launch 로 전달)
        _include("humanoid_bringup", CONTROL_LAUNCH,
                 condition=IfCondition(use_control),
                 launch_arguments={"use_teleop": use_teleop,
                                   # 킥 러너(walkkick_v1) 실행 여부.
                                   # match 는 true 로 넘긴다. 이게 false 면
                                   # visual_servo 가 /kick_trigger 를 쏴도
                                   # 듣는 노드가 없어 FSM 이 KICK_WAIT 에서
                                   # 멈춘다(= 경기 중 킥이 안 나감).
                                   "use_kick": use_kick,
                                   "kick_resume_hold": kick_resume_hold,
                                   "kick_ball_x": kick_ball_x,
                                   "kick_cmd_vx": kick_cmd_vx,
                                   "kick_duration": kick_duration,
                                   "kick_start_hold": kick_start_hold,
                                   "kick_strict_gate": kick_strict_gate,
                                   # [2026-08 CPU 최적화]
                                   "gravity_source":
                                       LaunchConfiguration("gravity_source"),
                                   "fall_manager_imu_topic":
                                       LaunchConfiguration(
                                           "fall_manager_imu_topic")}),

        # 3) 인지 레이어
        _include("humanoid_perception", PERCEPTION_LAUNCH,
                 condition=IfCondition(use_perception)),
    ])
