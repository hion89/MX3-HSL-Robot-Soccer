#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================================================
# control_teleop.launch.py  (humanoid_bringup)
# --------------------------------------------------------------------------
# control.launch.py(제어 레이어) + keyboard_teleop_node 를 함께 띄우되,
# 텔레옵 화면이 다른 노드 로그에 묻히지 않도록 두 가지를 처리한다:
#
#   (1) control 레이어 노드들의 출력을 'log'(파일)로 보낸다.
#       → 현재 터미널에 twist_mux/correction/actor/fall_manager 로그가
#         올라오지 않는다(로그는 ~/.ros/log/ 에 기록).
#   (2) keyboard_teleop_node 를 별도 터미널 창(xterm)에서 실행한다.
#       → 텔레옵의 상태줄(\r vx=.. vy=.. wz=..)이 단독 창에 깨끗하게 뜨고,
#         stdin(키 입력)이 그 창에 정상적으로 연결된다.
#
# [필요 패키지] xterm
#   설치: sudo apt install xterm
#   (xterm 이 없으면 텔레옵 창이 안 뜬다 — 아래 대안 참고)
#
# [대안 — launch 한 방이 싫거나 xterm 이 없을 때]
#   터미널1:  ros2 launch humanoid_bringup control.launch.py node_output:=log
#   터미널2:  ros2 run humanoid_policy_runner keyboard_teleop_node \
#                 --ros-args -r cmd_vel:=cmd_vel_teleop
#   (이 방식이 stdin 측면에서 가장 확실하다)
#
# 인자:
#   teleop_term (xterm -e) : 텔레옵을 띄울 터미널 prefix. gnome-terminal 을
#                            쓰려면 'gnome-terminal --' 등으로 바꿀 수 있다.
# ==========================================================================

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    teleop_term = LaunchConfiguration("teleop_term")
    # 보행+킥 통합 노드 경로 인자 (control.launch.py 로 그대로 전달)
    walk_actor_path = LaunchConfiguration("walk_actor_path")
    residual_actor_path = LaunchConfiguration("residual_actor_path")
    reference_csv_path = LaunchConfiguration("reference_csv_path")

    # control.launch.py 를 include 하되, 노드 출력을 'log' 로 돌려
    # 이 터미널을 조용하게 만든다. (control 내부의 use_teleop 은 false 로 두고
    # 텔레옵은 아래에서 xterm 으로 따로 띄운다.)
    control_path = os.path.join(
        get_package_share_directory("humanoid_bringup"),
        "launch", "control.launch.py",
    )
    # 정책 기본 경로(노드 기본값과 동일, 패키지 models/). control.launch.py 로 전달.
    _pol_models = os.path.join(
        get_package_share_directory("humanoid_policy_runner"), "models")
    control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(control_path),
        launch_arguments={
            "node_output": "log",
            "use_teleop": "false",
            "robot": LaunchConfiguration("robot"),   # robot_params.yaml 섹션 선택
            "walk_actor_path": walk_actor_path,
            "residual_actor_path": residual_actor_path,
            "reference_csv_path": reference_csv_path,
        }.items(),
    )

    # 텔레옵을 별도 터미널 창에서 실행.
    #   prefix=teleop_term → 예: 'xterm -e' 가 노드 실행 명령 앞에 붙어
    #     새 창에서 keyboard_teleop_node 가 돌아간다.
    #   emulate_tty=True   → 출력 버퍼링/제어문자가 tty 처럼 동작.
    #   remap cmd_vel → cmd_vel_teleop : twist_mux 의 teleop 입력으로 연결.
    teleop_node = Node(
        package="humanoid_policy_runner",
        executable="keyboard_teleop_node",
        name="keyboard_teleop_node",
        output="screen",
        emulate_tty=True,
        prefix=teleop_term,
        remappings=[("cmd_vel", "cmd_vel_teleop")],
    )

    return LaunchDescription([
        DeclareLaunchArgument("teleop_term", default_value="xterm -e"),
        DeclareLaunchArgument("robot", default_value=""),  # 비면 hostname 앞부분
        DeclareLaunchArgument("walk_actor_path",
                              # 2026-08: policy.onnx → walk_policy.onnx (v1s 러너)
                              # control.launch.py 로 명시 전달되므로 여기 기본값이 우선.
                              default_value=os.path.join(_pol_models, "walk_policy.onnx")),
        DeclareLaunchArgument("residual_actor_path",
                              default_value=os.path.join(_pol_models, "residual_kick_power_policy.onnx")),
        DeclareLaunchArgument("reference_csv_path", default_value=""),  # 비우면 ~/trajectories 최신
        control,
        teleop_node,
    ])