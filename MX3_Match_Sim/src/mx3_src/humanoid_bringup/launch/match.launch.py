#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================================================
# match.launch.py  (humanoid_bringup)  — 경기 한 방 기동(평면 구성)
# --------------------------------------------------------------------------
# 아래 leaf 런치들을 "지정한 순서 그대로" 하나씩 include 한다.
# 중간 그룹 런치(perception_stack 등)를 거치지 않는 평면(flat) 구성이라,
# 손으로 아래 명령들을 순서대로 친 것과 결과가 동일하다.
# 인자도 일부러 거의 안 넘긴다(= 각 leaf 의 기본값을 그대로 사용) → 상위/하위
# 런치 인자 '이름 충돌'이 원천적으로 생길 수 없다. 조건 토글도 두지 않는다
# (켜면 전부 켜짐 — 디버그로 일부만 띄울 땐 해당 leaf 를 직접 실행).
#
#   기동 순서(= 손으로 치던 명령 순서 그대로)
#   ----------------------------------------------------------------------
#   1) humanoid_bringup        bringup.launch.py
#        · 하드웨어(dynamixel_system) + control(twist_mux/보행 actor/fall_manager)
#        · bringup 기본값 use_perception=false → 인지는 3)이 전담(카메라/RSP 중복 없음)
#   2) humanoid_localization   ekf_localization.launch.py
#        · 비전+보행오도 EKF 융합 → map→odom TF
#   3) humanoid_perception     perception_bringup.launch.py
#        · robot_state_publisher + D405 + YOLO3D + 공 map 투영 (인지 스택 전체)
#          ※ perception_bringup 이 enable_yolo/enable_ball_map(기본 true)로
#            yolo_3d.launch.py·ball_map.launch.py 를 내부 include 하므로,
#            여기서 이 한 줄만 부르면 검출/투영까지 전부 뜬다.
#   4) humanoid_localization   pose_goal.launch.py        (avoid_enable:=true)
#        · 공통 go-to-pose 컨트롤러(+장애물 회피 합성)
#   5) humanoid_perception     depth_obstacle.launch.py
#        · 단거리 깊이 장애물 → /obstacle/avoid_hint
#   6) humanoid_serial_bridge  neck_ball_tracker.launch.py
#        · 시선(목) 중재자 — /neck/mode_request/* 소비
#   7) humanoid_localization   ball_follow.launch.py
#        · 공 rough approach (/pose_goal 발행)
#   8) humanoid_localization   vision_servo.launch.py
#        · 최종 정렬 + 킥 (cmd_vel_behavior)
#   9) humanoid_game_controller game_controller.launch.py   [추가]
#        · UDP 3838 수신 → /game_state, game_stop_lock, 개인상태 송신
#  10) humanoid_strategy       strategy.launch.py            [추가]
#        · role_assigner + strategy_fsm(스킬 enable 주인)
#  11) humanoid_strategy       team_comm.launch.py           [추가]
#        · 팀 통신(UDP broadcast) → /team_comm/state
#
# [robot/team/player 자동 결정]
#   9~11 의 leaf 런치가 각자 hostname(robo02-desktop→robo02) + robot_params.yaml
#   로 team_number/player_number 를 알아서 정한다. 그래서 여기서 따로 안 넘겨도
#   각 로봇에서 그냥 띄우면 그 로봇 번호로 적용된다.
#
# [단일 머신 launch 한계]
#   ros2 launch 는 '기동 순서'만 보장하고 '준비 완료'는 보장 안 한다. 다만 각
#   노드가 입력 토픽/TF 미수신 시 graceful wait(에러 아님)라, 위 순서로 띄우면
#   늦게 준비되는 상류(TF/검출)를 하류가 조용히 기다렸다가 자연히 동작한다.
#
# [실행]
#   ros2 launch humanoid_bringup match.launch.py
#
# [참고]
#   · match 는 ekf include 에 'rviz':'false' 를 넘겨 RViz 없이(헤드리스) 뜬다.
#     RViz 로 디버그하려면 그 값을 'true' 로 바꾸거나, ekf_localization.launch.py
#     를 직접 rviz:=true 로 실행한다(ekf 자체 기본값은 rviz:=true).
#   · pose_goal/ball_follow/vision_servo 는 leaf 기본 auto_start:=false 라
#     대기(IDLE)로만 뜬다 — 실제 enable 은 strategy_fsm 이 건다.
# ==========================================================================

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


# ══════════════════════════════════════════════════════════════════════════
#  ★★★ 경기장 규격 — 여기 한 줄만 바꾸면 전체 스택이 함께 전환된다 ★★★
# --------------------------------------------------------------------------
#   "s_field"  : 대회 9 x 6 (HSL 2026 small division). match 의 기본값.
#   "practice" : 연습장 3.9 x 2.6 (인조잔디 5x3 안의 라인)
#
#   이 값은 아래 두 곳에 '같은 값'으로 전달된다:
#     · ekf_localization.launch.py → 측위 랜드마크 지도 / 재진입 좌표 / RViz 라인
#     · strategy.launch.py        → 골대·필드 경계·포메이션 좌표 자동 축척
#
#   ★ 둘이 어긋나면 조용히 오동작한다 ★
#     측위는 연습장 지도로 수렴하는데 전략은 9x6 좌표(x=±4.5)를 목표로 잡아,
#     로봇이 필드 밖으로 걸어나간다. 그래서 인자로 따로 받지 않고 이 상수
#     하나에서 두 곳으로 뿌린다(= 어긋날 수가 없다).
#
#   ※ match 가 launch 인자를 두지 않는 원래 설계(상위/하위 인자 이름 충돌
#     원천 차단)를 유지하기 위해, 런타임 인자가 아니라 모듈 상수로 둔다.
FIELD_PROFILE = "s_field"
# ══════════════════════════════════════════════════════════════════════════


def _include(pkg: str, launch_file: str, launch_arguments=None):
    """지정 패키지 share/launch 에서 launch 파일을 찾아 include 하는 헬퍼.

    launch_arguments 에는 상위 인자를 끌어다 넘기지 않고 '리터럴 문자열'만
    넘긴다 → 상위/하위 런치 인자 이름 충돌 자체가 불가능.
    """
    path = os.path.join(
        get_package_share_directory(pkg), "launch", launch_file)
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(path),
        launch_arguments=(launch_arguments or {}).items(),
    )


def generate_launch_description():
    return LaunchDescription([
        # 1) 본체 — 하드웨어 + control (bringup 기본값 use_perception=false)
        #    ★ use_kick=true — 킥 러너(actor_onnx_walkkick_v1_runner)를 켠다 ★
        #      경기에서 킥은 이 경로로만 나간다:
        #        strategy_fsm(ALIGN_KICK) → visual_servo_kick_node 정렬
        #          → 실측 공위치를 킥 노드에 push → /kick_trigger
        #          → walking 러너 양보 → 킥 → /kick_done
        #          → walking 복귀 + FSM post_kick_settle(3.0s, cmd_vel 침묵)
        #      이게 false 면 /kick_trigger 구독자가 없어 /kick_done 이 영영
        #      오지 않고, FSM 이 ALIGN_KICK 하드 래치에 갇힌다(경기 중 킥 0회).
        _include("humanoid_bringup", "bringup.launch.py",
                 launch_arguments={"use_kick": "true"}),

        # 2) 로컬라이제이션 — EKF (map→odom TF)
        #    ★ match = 대회이므로 경기장 규격을 s_field(9x6, HSL 2026 small)로 고정한다.
        #      (리터럴 전달 — pose_goal 의 avoid_enable 과 동일 패턴이라 인자 충돌 없음)
        #      이 한 줄이 없으면 ekf 가 기본값 practice(3.9x2.6)로 떠서, 실제 9x6
        #      필드에서 랜드마크 지도가 안 맞아 위치인식이 수렴하지 않는다.
        #    ※ 연습장에서 full-stack 리허설을 할 땐 이 값만 "practice" 로 바꾸면 된다.
        #    ★ rviz:false — match 는 헤드리스가 기본. RViz 디버그가 필요하면 "true" 로
        #      바꾸거나, ekf_localization.launch.py 를 직접 rviz:=true 로 띄운다.
        _include("humanoid_localization", "ekf_localization.launch.py",
                 launch_arguments={"field_profile": FIELD_PROFILE,
                                   "rviz": "false"}),

        # 3) 인지 스택 전체 — RSP + 카메라 + YOLO3D + 공 map 투영
        #    (perception_bringup 이 yolo_3d·ball_map 을 내부 include. 기본 전부 ON)
        _include("humanoid_perception", "perception_bringup.launch.py"),

        # 4) 공통 위치이동 컨트롤러 — 장애물 회피 합성 ON
        _include("humanoid_localization", "pose_goal.launch.py",
                 launch_arguments={"avoid_enable": "true"}),

        # 5) 단거리 깊이 장애물 검출
        _include("humanoid_perception", "depth_obstacle.launch.py"),

        # 6) 시선(목) 중재자
        _include("humanoid_serial_bridge", "neck_ball_tracker.launch.py"),

        # 7) 공 rough approach
        _include("humanoid_localization", "ball_follow.launch.py"),

        # 8) 최종 정렬 + 킥
        _include("humanoid_localization", "vision_servo.launch.py"),

        # 9) GameController (추가)
        _include("humanoid_game_controller", "game_controller.launch.py"),

        # 10) 전략 (추가)
        #     ★ EKF 와 '같은' FIELD_PROFILE 을 넘긴다 ★
        #       strategy.yaml 좌표는 s_field(9x6) 기준이라, practice 로 띄우면
        #       골대/필드경계/포메이션 좌표가 자동으로 x0.4333 축척된다.
        #       이걸 안 넘기면 측위만 연습장이고 전략은 9x6 좌표를 써서
        #       로봇이 필드 밖(x=±4.5)을 목표로 걸어나간다.
        _include("humanoid_strategy", "strategy.launch.py",
                 launch_arguments={"field_profile": FIELD_PROFILE}),

        # 11) 팀 통신 (추가)
        _include("humanoid_strategy", "team_comm.launch.py"),
    ])
