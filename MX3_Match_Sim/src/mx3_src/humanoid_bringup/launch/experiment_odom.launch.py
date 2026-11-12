#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================================================
# experiment_odom.launch.py  (humanoid_bringup)
# --------------------------------------------------------------------------
# 오도메트리 캘리브레이션 실험 전용 런치. 경기/개발용이 아니다.
#
#   control.launch.py  +  keyboard_teleop_node(구간 마커)  +  실험 로거
#
# [무엇을 재는 실험인가]
#   길이를 아는 직선 구간(예: 5m)을 여러 속도로 주행하며 소요 시간을 잰다.
#   그동안 /joint_states 를 함께 기록해 두면, 같은 데이터 하나로
#     (1) 지금의 cmd_vel 기반 calib 테이블 재피팅
#     (2) 관절 기구학 기반 오도메트리
#   를 모두 오프라인에서 계산해 비교할 수 있다.
#   → 어느 쪽이 정확한지 실시간 노드를 만들기 전에 숫자로 판정한다.
#
# [조작]
#   W/S/A/D/Q/E : 속도 증감 (control_teleop.launch.py 와 동일)
#   ,           : 측정 구간 시작   ← 출발선 A 를 지나는 순간
#   .           : 측정 구간 종료   ← 도착선 B 를 지나는 순간
#   x           : 구간 취소 (잘못 눌렀을 때)
#   SPACE       : 급정지
#
#   ★ 조주 구간을 두십시오 ★
#   A 를 지날 때 이미 정속이어야 한다. 출발 가속이 측정 구간에 포함되면
#   실제보다 낮은 속도가 나온다(기존 4m/17초 측정이 낮게 나온 원인).
#     [출발] --1.5m 조주-- [A] --5m 측정-- [B] --여유--
#
# [속도를 정확히 지정하는 법]
#   키보드 텔레옵은 명령값이 이산적이라 스틱과 달리 값이 정확히 알려진다.
#   vx_pos_min 과 vx_pos_max 를 같은 값으로 두면 W 한 번에 그 값으로
#   고정되어, 회차마다 명령이 흔들리지 않는다.
#
#     ros2 launch humanoid_bringup experiment_odom.launch.py \
#         vx_pos_min:=0.30 vx_pos_max:=0.30 label:=fwd_030
#
#   여러 속도를 한 세션에 섞으려면 min/max/step 을 열어 두면 된다.
#   어느 쪽이든 실제 명령값은 로거 CSV 의 cmd_vx 컬럼에 남으므로,
#   분석은 화면에 뜬 숫자가 아니라 CSV 를 근거로 한다.
#
# [출력]
#   ~/odom_experiments/<label>_<robot>_<시각>_raw.csv
#       /joint_states 매 프레임(50Hz) 한 줄. 관절 위치·속도, IMU, cmd,
#       odom, gait 위상이 한 시간축에 정렬돼 있다.
#   ~/odom_experiments/<label>_<robot>_<시각>_segments.csv
#       측정 구간당 한 줄. 소요 시간과 현재 odom 이 말하는 이동거리.
#       → 실험 직후 이 파일만 열어도 1 차 결과가 보인다.
#
# [RViz / Foxglove]
#   /experiment/viz (MarkerArray) 로 구간 시작점·주행 궤적·상태 텍스트를
#   발행한다. 궤적이 휘었으면 그 회차는 그 자리에서 다시 하면 된다.
#
# [사전 확인]
#   1) 로봇을 매단 채로 한 번 띄워 jtop 으로 부하를 본다. 로거가 보행에
#      영향을 주면 데이터가 오염된다 — 이 실험에서 제일 나쁜 경우다.
#   2) `,` 를 눌렀을 때 터미널에 REC 표시가 뜨는지 확인한다.
#   3) 방향키를 눌러도 로봇이 돌지 않는지 확인한다(2026-08-28 수정분).
#
# 인자:
#   robot ("")                : robot_params.yaml 섹션 (비면 hostname)
#   label ("run")             : 출력 파일명 라벨. 세션 성격을 적는다.
#                               예: fwd_030, bwd_030, square_check
#   output_dir (~/odom_experiments)
#   teleop_term ("xterm -e")  : 텔레옵을 띄울 터미널 prefix
#   vx_pos_min / vx_pos_max / vx_step : 전진 명령 범위
#   vx_neg_min / vx_neg_max   : 후진 명령 범위
#   wz_min / wz_max / vy_min / vy_max : 회전·횡이동 범위
#   segment_id_start (1)      : 구간 번호 시작값. 세션을 나눠 찍을 때
#                               번호가 겹치지 않도록 올려서 실행한다.
#   log_outside_segments (true): 구간 밖도 기록할지
#   use_kick (false)          : 실험 중엔 킥이 필요 없다
#   imu_topic (/imu/data_50hz): ★ 오도메트리 노드가 실제로 쓰는 토픽 ★
#                               200Hz 원본을 쓰면 오프라인 재현이 실제와
#                               달라진다.
# ==========================================================================

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    robot = LaunchConfiguration("robot")
    label = LaunchConfiguration("label")
    output_dir = LaunchConfiguration("output_dir")
    teleop_term = LaunchConfiguration("teleop_term")
    imu_topic = LaunchConfiguration("imu_topic")
    segment_id_start = LaunchConfiguration("segment_id_start")
    log_outside = LaunchConfiguration("log_outside_segments")
    use_kick = LaunchConfiguration("use_kick")

    vx_pos_min = LaunchConfiguration("vx_pos_min")
    vx_pos_max = LaunchConfiguration("vx_pos_max")
    vx_neg_min = LaunchConfiguration("vx_neg_min")
    vx_neg_max = LaunchConfiguration("vx_neg_max")
    vx_step = LaunchConfiguration("vx_step")
    vy_min = LaunchConfiguration("vy_min")
    vy_max = LaunchConfiguration("vy_max")
    wz_min = LaunchConfiguration("wz_min")
    wz_max = LaunchConfiguration("wz_max")

    walk_actor_path = LaunchConfiguration("walk_actor_path")
    residual_actor_path = LaunchConfiguration("residual_actor_path")
    reference_csv_path = LaunchConfiguration("reference_csv_path")

    _pol_models = os.path.join(
        get_package_share_directory("humanoid_policy_runner"), "models")

    # ── control 레이어 ──────────────────────────────────────────
    #   node_output=log 로 터미널을 조용히 유지한다. 실험 중에는 로거의
    #   구간 결과 로그만 보여야 놓치지 않는다.
    control_path = os.path.join(
        get_package_share_directory("humanoid_bringup"),
        "launch", "control.launch.py")
    control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(control_path),
        launch_arguments={
            "node_output": "log",
            "use_teleop": "false",     # 텔레옵은 아래에서 xterm 으로
            "robot": robot,
            "use_kick": use_kick,
            "walk_actor_path": walk_actor_path,
            "residual_actor_path": residual_actor_path,
            "reference_csv_path": reference_csv_path,
        }.items(),
    )

    # ── 키보드 텔레옵 (구간 마커 포함) ──────────────────────────
    #   별도 xterm 창에서 띄운다. stdin 을 raw 모드로 잡아야 하고,
    #   상태줄이 다른 로그에 묻히면 안 되기 때문이다.
    #   cmd_vel → cmd_vel_teleop 으로 remap 해 twist_mux 를 태운다.
    #   ★ marker_topic 은 절대경로라 이 remap 에 끌려가지 않는다 ★
    teleop_node = Node(
        package="humanoid_policy_runner",
        executable="keyboard_teleop_node",
        name="keyboard_teleop_node",
        output="screen",
        emulate_tty=True,
        prefix=teleop_term,
        remappings=[("cmd_vel", "cmd_vel_teleop")],
        parameters=[{
            # LaunchConfiguration 은 문자열이므로 노드가 선언한 double/int
            # 파라미터와 타입을 맞추려면 ParameterValue 로 감싸야 한다.
            # 안 감싸면 InvalidParameterTypeException 으로 즉사한다.
            "vx_pos_min": ParameterValue(vx_pos_min, value_type=float),
            "vx_pos_max": ParameterValue(vx_pos_max, value_type=float),
            "vx_neg_min": ParameterValue(vx_neg_min, value_type=float),
            "vx_neg_max": ParameterValue(vx_neg_max, value_type=float),
            "vx_step": ParameterValue(vx_step, value_type=float),
            "vy_min": ParameterValue(vy_min, value_type=float),
            "vy_max": ParameterValue(vy_max, value_type=float),
            "wz_min": ParameterValue(wz_min, value_type=float),
            "wz_max": ParameterValue(wz_max, value_type=float),
            "marker_topic": "/experiment/marker",
            "segment_id_start": ParameterValue(segment_id_start,
                                               value_type=int),
        }],
    )

    # ── 실험 로거 ───────────────────────────────────────────────
    #   실시간 제약이 없는 별도 노드다. 죽어도 로봇에 영향이 없고,
    #   실험 안 할 때는 이 런치를 안 쓰면 그만이다.
    #   output=screen 으로 두어 구간 종료 시 소요 시간이 이 터미널에
    #   바로 뜨게 한다(텔레옵 xterm 과 별개로 기록이 남는다).
    logger_node = Node(
        package="humanoid_utils",
        executable="experiment_logger_node",
        name="experiment_logger_node",
        output="screen",
        emulate_tty=True,
        parameters=[{
            "output_dir": output_dir,
            "label": label,
            "robot": robot,
            "imu_topic": imu_topic,
            "marker_topic": "/experiment/marker",
            "log_outside_segments": ParameterValue(log_outside,
                                                   value_type=bool),
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "robot", default_value="",
            description="robot_params.yaml 섹션. 비우면 hostname 앞부분."),
        DeclareLaunchArgument(
            "label", default_value="run",
            description="출력 파일명 라벨. 세션 성격을 적는다 "
                        "(fwd_030, bwd_030, square_check 등). "
                        "나중에 학습용과 검증용을 파일 단위로 나눌 때 쓴다."),
        DeclareLaunchArgument(
            "output_dir", default_value="~/odom_experiments",
            description="CSV 출력 디렉터리. NVMe 경로를 권장한다."),
        DeclareLaunchArgument(
            "teleop_term", default_value="xterm -e",
            description="텔레옵을 띄울 터미널 prefix. "
                        "gnome-terminal 을 쓰면 'gnome-terminal --'."),
        DeclareLaunchArgument(
            "imu_topic", default_value="/imu/data_50hz",
            description="★ walk_odometry 가 실제로 구독하는 토픽과 같아야 "
                        "한다 ★ 200Hz 원본(/imu/data)으로 기록하면 "
                        "오프라인 재현이 실제 노드와 달라진다."),
        DeclareLaunchArgument(
            "segment_id_start", default_value="1",
            description="구간 번호 시작값. 세션을 나눠 찍을 때 번호가 "
                        "겹치지 않도록 올려서 실행한다(예: 2세션은 101)."),
        DeclareLaunchArgument(
            "log_outside_segments", default_value="true",
            description="측정 구간 밖도 기록할지. false 면 파일이 작아지지만 "
                        "구간 사이에 무슨 일이 있었는지 볼 수 없다."),
        DeclareLaunchArgument(
            "use_kick", default_value="false",
            description="실험 중엔 킥 러너가 필요 없다. 띄우면 /joint_goals "
                        "소유권 전환이 데이터에 섞일 수 있다."),

        # ── 명령 범위 ──
        #   min 과 max 를 같은 값으로 두면 W 한 번에 그 값으로 고정된다.
        #   회차마다 명령이 흔들리지 않아 실험에 유리하다.
        DeclareLaunchArgument(
            "vx_pos_min", default_value="0.15",
            description="전진 최소 명령. max 와 같게 두면 단일 속도 고정."),
        DeclareLaunchArgument("vx_pos_max", default_value="0.50"),
        DeclareLaunchArgument("vx_neg_min", default_value="0.10"),
        DeclareLaunchArgument("vx_neg_max", default_value="0.30"),
        DeclareLaunchArgument("vx_step", default_value="0.05"),
        DeclareLaunchArgument("vy_min", default_value="0.10"),
        DeclareLaunchArgument("vy_max", default_value="0.30"),
        DeclareLaunchArgument("wz_min", default_value="0.50"),
        DeclareLaunchArgument("wz_max", default_value="1.50"),

        DeclareLaunchArgument(
            "walk_actor_path",
            default_value=os.path.join(_pol_models, "walk_policy.onnx")),
        DeclareLaunchArgument(
            "residual_actor_path",
            default_value=os.path.join(
                _pol_models, "residual_kick_power_policy.onnx")),
        DeclareLaunchArgument("reference_csv_path", default_value=""),

        control,
        teleop_node,
        logger_node,
    ])
