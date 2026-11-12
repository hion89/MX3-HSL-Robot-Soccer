#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================================================
# control_joypad.launch.py  (humanoid_bringup)
# --------------------------------------------------------------------------
# control.launch.py(제어 레이어) + joy_node + joypad_teleop_node 를 함께 띄운다.
# control_teleop.launch.py 의 게임패드 버전이다. 키보드와 달리 stdin 이 필요 없어
# xterm 같은 별도 터미널 창이 필요 없다(그래서 더 단순하다).
#
# [2026-08 정책 교체 / 개명]
#   actor 가 walking 러너(구 v1s 러너)로 바뀌면서 조이패드도 전용 노드로
#   교체됐고, 2026-08-18 개명으로 이름이 아래처럼 정리됐다.
#     actor_onnx_v1s_runner  → actor_onnx_walking_runner
#     joypad_teleop_v1s_node → joypad_teleop_node
#   v1s_joypad_test.launch.py(벤치 전용) 의 기능을 그대로 살리되, 벤치와 달리
#   /cmd_vel 직결이 아니라 cmd_vel_teleop → twist_mux 를 태운다.
#
# [2026-08-24 보행 주파수 가변화]
#   보행 정책이 교체되면서 주파수가 고정이 아니라 1.7~2.4 Hz 범위에서
#   가변이 되었다. 그래서 D-pad 상하로 주파수를 실제로 바꿀 수 있게
#   기본값을 아래처럼 열었다.
#     gait_freq_min : 2.0 → 1.7   (이전에는 min=max=2.0 으로 잠가 뒀다)
#     gait_freq_max : 2.0 → 2.4
#     gait_freq_step: 0.1 (하드코딩 → launch 인자로 노출)
#   기본 시작값 gait_freq 는 2.0 을 유지한다. 0.1 스텝으로 1.7 과 2.4
#   양쪽 한계에 정확히 도달하는 값이고, 이전 동작과 연속성이 있다.
#
#   ★ 학습 분포를 벗어나면 성능 보장이 안 된다 ★
#   min/max 는 "정책이 학습한 주파수 분포" 그 자체여야 한다. 정책을 또
#   교체할 때는 이 두 값을 학습 cfg 와 다시 맞출 것. 러너
#   (actor_onnx_walking_runner.cb_gait_frequency)는 분포 밖 값을 막지
#   않고 경고만 하므로, 실질적인 안전장치는 여기 min/max 뿐이다.
#
#   ※ policy_variant 는 v0_2 기본값을 그대로 둔다. 러너 코드상 v0_2 와
#     v1s 는 gait 관측 3 차원 / 주파수 처리 경로가 완전히 동일하고
#     (분기는 v0_1 · v1 에만 있다), 주파수를 속도 명령이 아니라 외부
#     명령으로 받는 쪽이다. 즉 라벨만 과거 이름이고 동작은 가변 주파수로
#     이미 맞다. 다만 관측 정규화 스케일(control.launch.py 의
#     gait_frequency_scale=2.0)은 학습 cfg 와 일치해야 하므로,
#     새 정책이 다른 스케일로 학습됐다면 그쪽도 같이 고쳐야 한다.
#
# [명령 흐름]
#   8BitDo 패드 → joy_node → /joy → joypad_teleop_node → cmd_vel_teleop
#     → twist_mux(prio 100) → /cmd_vel_raw → drift보정 → /cmd_vel → actor
#   같은 노드가 /gait_mode, /gait_frequency 를 latched 로 발행해 actor 에 직결된다.
#
# [조작법 — 8BitDo Ultimate 2 (XInput)]
#   왼쪽 스틱 상하 : 전진/후진 (vx)
#   왼쪽 스틱 좌우 : 횡이동 (vy)
#   오른쪽 스틱 좌우 : 회전 (wz)
#   D-pad 좌우     : gait_mode 순환 (auto ↔ march)
#   D-pad 상하     : gait_frequency 증감 (1.7~2.4 Hz, 0.1 Hz 단위)
#                    한 번 누르면 1 스텝, 계속 누르고 있으면 0.25s 마다 반복.
#                    (반복 주기는 노드 파라미터 gait_freq_repeat_s 기본값)
#   LT / RT        : 회전 부스트 (당기고 있는 동안만 wz 상한 ↑)
#                    2 단 구성: 평소 wz_max(1.5) → 홀드 시 wz_max_boost(3.0)
#                    놓으면 즉시 복귀한다(토글 아님).
#                    ★ 기동 후 각 트리거를 한 번씩 당겼다 놓아야 무장된다 ★
#                      (미조작 시 축을 0.0 으로 보고하는 드라이버 대응)
#   A 버튼         : 킥 트리거 (/kick_trigger)
#   B 버튼         : 긴급 정지 (auto 로 되돌리고 속도 0)
#   Y 버튼         : 주파수 기본값(2.0 Hz) 복귀
#
# [첫 테스트는 로봇을 매단 상태에서]
#   1) 스틱을 놓은 auto 상태로 시작 → 축/버튼 매핑 확인 (움직이지 않아야 정상)
#   2) D-pad 우 → march 로 전환 → 제자리 걷기 확인
#   3) B 버튼 → 두 발이 모이며 멈추는지 확인
#   4) D-pad 우 → auto → 왼쪽 스틱 살짝 밀어 전진
#
# [사전 준비 — OS/하드웨어]
#   1) 패드를 XInput 모드로 켠다(아무 버튼도 안 누르고 전원 ON = XInput).
#      2.4GHz 동글 연결을 권장(지연/안정성 최상). 유선 USB 도 가능.
#   2) 인식 확인:  lsusb          (8BitDo, VID 2dc8 확인)
#                  ls /dev/input/js*   ← js0 가 보이면 OK
#   3) js0 이 안 뜨면(JetPack 6.x 커널 5.15 에서 신형 PID 미인식 가능):
#      xpad 드라이버를 강제 바인딩하는 udev 규칙을 추가한다(README/설명 참조).
#   4) 매핑 검증:  ros2 run joy joy_node   후   ros2 topic echo /joy
#      로 axes/buttons 인덱스와 부호를 확인하고, 필요하면 joypad_teleop_node
#      파라미터(axis_*, invert_*, button_*)로 맞춘다.
#
# [대안 — launch 없이 2터미널로 직접]
#   터미널1: ros2 launch humanoid_bringup control.launch.py node_output:=log
#   터미널2: ros2 run joy joy_node &
#            ros2 run humanoid_policy_runner joypad_teleop_node --ros-args \
#                -p cmd_vel_topic:=cmd_vel_teleop \
#                -p axis_vx:=1 -p axis_vy:=0 -p axis_wz:=2 -p button_kick:=0
#            (cmd_vel 은 remap 이 아니라 cmd_vel_topic 파라미터로 지정한다)
#
# 인자:
#   robot ("")               : robot_params.yaml 섹션(비면 hostname 앞부분 자동)
#   joy_dev (0)              : joy_node device_id (여러 입력장치가 있으면 조정)
#   joy_deadzone (0.05)      : joy_node 자체 데드존(raw). 스틱 중립 오차 제거용.
#                              joy_node 는 이 값을 잘라낸 뒤 다시 ±1.0 으로
#                              재정규화하므로 /joy 의 조작 범위는 줄지 않는다.
#   teleop_deadzone (0.20)   : joypad_teleop_node 의 속도 매핑 데드존.
#                              ★ 실제로 "스틱을 얼마나 밀어야 움직이는가"를
#                                결정하는 값. joy_deadzone 과 별개다.
#                              (2026-08-18 이전에는 joy_deadzone 을 이쪽에도
#                               같이 넘겨 0.05 가 적용되고 있었다 — 버그)
#   wz_max (1.5)             : 최대 회전속도 [rad/s]. 오른쪽 스틱을 끝까지
#                              밀었을 때 나가는 명령값. (2026-08-28 인자화)
#   wz_min (0.5)             : 데드존 통과 직후 최소 회전속도 [rad/s].
#                              0 이 아니면 릴리스 시 계단이 남는다.
#   wz_max_boost (3.0)       : LT/RT 홀드 중 회전속도 상한 [rad/s].
#                              0.0 이면 부스트 비활성.
#   axis_wz_boost_l (4)      : LT 축 인덱스 (-1=미사용). ★실측 필수★
#   axis_wz_boost_r (5)      : RT 축 인덱스 (-1=미사용). ★실측 필수★
#   wz_boost_rest (1.0)      : 트리거를 놓았을 때의 축 값 (+1.0 / -1.0)
#   wz_boost_on (0.60)       : 부스트 ON 임계 (당긴 정도 0~1)
#   wz_boost_off (0.40)      : 부스트 OFF 임계. on 보다 작아야 한다.
#   enable_button (-1)       : 데드맨 스위치 버튼 인덱스(-1=비활성, 예: 5=RB)
#   button_kick (0)          : 킥 버튼 인덱스(기본 A)
#   gait_freq (2.0)          : 시작 보행 주파수 [Hz]. Y 버튼 복귀값도 이 값.
#   gait_freq_min (1.7)      : D-pad 하 로 내릴 수 있는 하한 [Hz]
#   gait_freq_max (2.4)      : D-pad 상 으로 올릴 수 있는 상한 [Hz]
#   gait_freq_step (0.1)     : D-pad 1 회 입력당 변화량 [Hz]
#   walk_actor_path / residual_actor_path / reference_csv_path : 모델/레퍼런스 경로
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
    joy_dev = LaunchConfiguration("joy_dev")
    joy_deadzone = LaunchConfiguration("joy_deadzone")
    teleop_deadzone = LaunchConfiguration("teleop_deadzone")
    slew_enable = LaunchConfiguration("slew_enable")
    accel_vx = LaunchConfiguration("accel_vx")
    decel_vx = LaunchConfiguration("decel_vx")
    accel_wz = LaunchConfiguration("accel_wz")
    decel_wz = LaunchConfiguration("decel_wz")
    # ── 킥(walkkick_v1) 인자 ──
    use_kick = LaunchConfiguration("use_kick")
    kick_ball_x = LaunchConfiguration("kick_ball_x")
    kick_cmd_vx = LaunchConfiguration("kick_cmd_vx")
    kick_duration = LaunchConfiguration("kick_duration")
    kick_strict_gate = LaunchConfiguration("kick_strict_gate")
    # ★ 두 hold 는 서로 다른 구간이다. 이름을 반드시 구분할 것 ★
    #   kick_start_hold : 트리거 직후 ~ 킥 시작. 급발진 방지(walkkick_v1).
    #   kick_end_hold   : 킥 종료 ~ 보행 복귀. 자세 안정화(v1s).
    kick_start_hold = LaunchConfiguration("kick_start_hold")
    kick_end_hold = LaunchConfiguration("kick_end_hold")
    enable_button = LaunchConfiguration("enable_button")
    button_kick = LaunchConfiguration("button_kick")
    deadzone_vy = LaunchConfiguration("deadzone_vy")
    # ── 회전(wz) 속도 범위 ──
    #   2026-08-28: 예전에는 joypad_node 파라미터 dict 안에 0.5 / 1.5 로
    #   하드코딩돼 있어서, 최대 회전속도를 바꾸려면 이 파일을 직접 고쳐야
    #   했다(런치 인자로 넘기면 선언이 없어 조용히 무시됨). 현장에서 자주
    #   만지는 값이므로 launch 인자로 끌어올린다.
    wz_min = LaunchConfiguration("wz_min")
    wz_max = LaunchConfiguration("wz_max")
    # ── 회전 부스트 (LT/RT 홀드) ──
    #   2026-08-28: 트리거를 당기고 있는 동안만 wz 상한을 올린다.
    #   패드 뒷면의 트리거 모드 스위치(홀 이펙트 ↔ 택타일)는 호스트로
    #   상태가 보고되지 않아 그 스위치 자체에는 묶을 수 없다. 그래서
    #   트리거의 '아날로그 축' 을 홀드 스위치처럼 쓴다.
    wz_max_boost = LaunchConfiguration("wz_max_boost")
    axis_wz_boost_l = LaunchConfiguration("axis_wz_boost_l")
    axis_wz_boost_r = LaunchConfiguration("axis_wz_boost_r")
    wz_boost_rest = LaunchConfiguration("wz_boost_rest")
    wz_boost_on = LaunchConfiguration("wz_boost_on")
    wz_boost_off = LaunchConfiguration("wz_boost_off")
    gait_mode = LaunchConfiguration("gait_mode")
    gait_freq = LaunchConfiguration("gait_freq")
    gait_freq_min = LaunchConfiguration("gait_freq_min")
    gait_freq_max = LaunchConfiguration("gait_freq_max")
    # D-pad 1 회 입력당 주파수 변화량. 이전에는 노드 파라미터에 0.1 로
    # 하드코딩돼 있었는데, 가변 범위가 열리면서 현장에서 조정할 여지가
    # 생겼으므로 launch 인자로 끌어올렸다.
    gait_freq_step = LaunchConfiguration("gait_freq_step")
    policy_variant = LaunchConfiguration("policy_variant")
    walk_actor_path = LaunchConfiguration("walk_actor_path")
    residual_actor_path = LaunchConfiguration("residual_actor_path")
    reference_csv_path = LaunchConfiguration("reference_csv_path")

    # 정책 기본 경로(노드 기본값과 동일, 패키지 models/). control.launch.py 로 전달.
    pol_models = os.path.join(
        get_package_share_directory("humanoid_policy_runner"), "models")

    # ── control 레이어 include ──
    #  node_output=log 로 돌려 이 터미널을 조용히 유지(twist_mux/actor 로그는 파일로).
    #  use_teleop=false: 키보드 텔레옵은 안 띄운다(여기선 게임패드 사용).
    control_path = os.path.join(
        get_package_share_directory("humanoid_bringup"),
        "launch", "control.launch.py")
    control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(control_path),
        launch_arguments={
            "node_output": "log",
            "use_teleop": "false",
            "robot": robot,
            "walk_actor_path": walk_actor_path,
            "residual_actor_path": residual_actor_path,
            "reference_csv_path": reference_csv_path,
            # 러너와 조이패드 노드가 같은 모드/주파수로 시작하도록 함께 넘긴다.
            # (조이패드가 /gait_mode 를 latched 로 쏘므로 값이 다르면 조이패드가 이긴다)
            "policy_variant": policy_variant,
            "gait_mode": gait_mode,
            "gait_freq": gait_freq,
            # 킥 '종료 후' 보행 복귀 hold (v1s). 킥 '전' hold 인
            # kick_start_hold 와 혼동하지 말 것.
            "kick_resume_hold": kick_end_hold,
            # 킥 러너는 control.launch.py 가 띄운다. 여기서는 켜기만 하고
            # 튜닝 인자를 그대로 넘긴다(같은 경로를 match 와 공유).
            "use_kick": use_kick,
            "kick_ball_x": kick_ball_x,
            "kick_cmd_vx": kick_cmd_vx,
            "kick_duration": kick_duration,
            "kick_start_hold": kick_start_hold,
            "kick_strict_gate": kick_strict_gate,
        }.items(),
    )

    # ── joy_node : 물리 패드 → sensor_msgs/Joy(/joy) ──
    #  device_id 로 어떤 입력장치를 읽을지 선택. autorepeat_rate 로 값이 안 바뀌어도
    #  주기적으로 /joy 를 재발행(텔레옵 노드도 자체 타이머가 있어 이중 안전).
    joy_node = Node(
        package="joy", executable="joy_node", name="joy_node",
        output="screen",
        parameters=[{
            # LaunchConfiguration 은 문자열이라, 노드가 선언한 int/float 파라미터와
            # 타입이 맞도록 ParameterValue 로 명시 변환한다(타입 불일치 시작 실패 방지).
            "device_id": ParameterValue(joy_dev, value_type=int),
            # joy_node 의 데드존은 스틱 물리 중립 오차(드리프트) 제거 전용이다.
            # 잘라낸 뒤 다시 ±1.0 으로 재정규화하므로(joy.cpp 의
            # scale_ = -1/(1-deadzone)/32767) /joy 의 범위는 줄지 않는다.
            # 실제 "얼마나 밀어야 움직이나"는 teleop_deadzone 이 결정한다.
            "deadzone": ParameterValue(joy_deadzone, value_type=float),
            "autorepeat_rate": 20.0,
        }],
    )

    # ── joypad_teleop_node : /joy → cmd_vel_teleop + /gait_mode + /gait_frequency ──
    #  2026-08 정책 교체(v1s 러너)에 맞춰 v0 조이패드 노드에서 교체했다.
    #  v0 노드 대비 추가되는 것:
    #    · D-pad 좌우 → gait_mode 순환 (auto ↔ march ↔ stop)
    #    · D-pad 상하 → gait_frequency 증감
    #    · B 버튼    → 즉시 stop 모드 (E-stop 대용)
    #
    #  [cmd_vel 은 remap 이 아니라 파라미터로 지정한다]
    #  이 노드는 cmd_vel_topic 파라미터로 발행 토픽을 정하므로, v0 처럼
    #  remappings 로 바꾸면 gait/kick 토픽까지 같이 끌려갈 수 있다.
    #  cmd_vel_teleop 으로 보내 twist_mux(teleop, prio 100) 를 태운다.
    #
    #  [gait_mode 는 latched 발행이다]
    #  이 노드가 TRANSIENT_LOCAL 로 /gait_mode 를 1회 발행하므로, 러너의
    #  gait_mode 파라미터보다 이 값이 최종적으로 이긴다. 그래서 control.launch.py
    #  로 넘기는 gait_mode 와 같은 값을 여기에도 넣어 둘을 일치시킨다.
    joypad_node = Node(
        package="humanoid_policy_runner", executable="joypad_teleop_node",
        name="joypad_teleop_node", output="screen",
        parameters=[{
            "cmd_vel_topic": "cmd_vel_teleop",

            # ── 8BitDo Ultimate 2 (블루투스, hid-generic) 실측 매핑 ──
            #   `ros2 topic echo /joy` 로 직접 확인한 값. 다른 패드/연결이면 여기만 바꾼다.
            #     axes[1] = 왼쪽 스틱 상하  → vx (위 = 전진)
            #     axes[0] = 왼쪽 스틱 좌우  → vy (좌 = +)
            #     axes[2] = 오른쪽 스틱 좌우 → wz (좌 = +)   ※ XInput 표준의 3이 아니라 2!
            #     axes[6/7] = D-pad 좌우/상하
            "axis_vx": 1,
            "axis_vy": 0,
            "axis_wz": 2,
            "axis_dpad_lr": 6,   # 좌우 → gait_mode 순환
            "axis_dpad_ud": 7,   # 상하 → gait_frequency 증감
            #   부호: joy_node 표준 규약(좌+/위+) 기준으로 전진=+, 좌횡=+, 좌회전=+ 가 되도록
            #   그대로 둔다. 실로봇에서 방향이 반대면 해당 invert_* 만 true 로 뒤집는다.
            "invert_vx": False,
            "invert_vy": False,
            "invert_wz": False,
            # ── 데드존 (2026-08-18 개편) ──
            #   ★ 여기에 joy_deadzone(0.05) 을 넘기던 것이 버그였다 ★
            #   joy_node 의 데드존은 스틱 중립 오차 제거용이고, 이 노드의
            #   데드존은 "스틱을 얼마나 밀어야 속도가 나가는가"를 정한다.
            #   서로 목적이 달라 같은 값을 쓸 이유가 없다. 전 축 0.20 통일.
            "deadzone": ParameterValue(teleop_deadzone, value_type=float),
            #   축별 개별 데드존. -1.0 이면 위 전역값(0.20)을 상속한다.
            #   예전에는 vy 만 0.35 로 크게 잡아 전진 중 대각선 새어듦을
            #   막았으나, vy_min=0 이 되면서 새어나가는 크기가 0 근처로
            #   작아졌으므로 전역 상속으로 통일했다. 실기에서 여전히
            #   옆으로 흐르면 deadzone_vy:=0.35 로 되돌리면 된다.
            "deadzone_vy": ParameterValue(deadzone_vy, value_type=float),
            # ── 축별 속도 범위 (2026-08-18 개편) ──
            #   *_min 을 0 으로 두는 것이 핵심이다. 0 이 아니면 데드존
            #   경계에서 출력이 0 → min 으로 수직 점프하고, 스틱을 놓을 때
            #   min → 0 으로 낙하한다. 이 계단이 velocity_commands 관측에
            #   그대로 들어가 릴리스 시 휘청임을 만든다.
            #   현재 정책은 저속에서도 잘 걸어 바닥이 필요 없다.
            "vx_pos_min": 0.0,
            "vx_neg_min": 0.0,
            "vy_min": 0.0,
            #   ⚠ wz 는 요청에 따라 0.50~1.50 을 유지한다. 즉 회전축에는
            #     0.50 rad/s 계단이 남는다. vx 를 0 으로 내린 뒤에도
            #     휘청임이 남으면 원인은 거의 이쪽이니 wz_min 을 0 으로.
            #   ★ 2026-08-28: 하드코딩 → launch 인자로 노출 ★
            #     LaunchConfiguration 은 문자열이므로 노드가 선언한 double
            #     파라미터와 타입을 맞추려면 ParameterValue(value_type=float)
            #     로 감싸야 한다. 안 감싸면 기동 시
            #     InvalidParameterTypeException 으로 이 노드만 즉사한다.
            "wz_min": ParameterValue(wz_min, value_type=float),
            "wz_max": ParameterValue(wz_max, value_type=float),
            # ── 회전 부스트 (LT/RT 홀드) ──
            #   트리거를 당기고 있는 동안만 상한이 wz_max_boost 로 오르고,
            #   놓으면 즉시 wz_max 로 복귀한다(토글 아님).
            #   ★ axis 인덱스는 반드시 `ros2 topic echo /joy` 로 실측할 것 ★
            "wz_max_boost": ParameterValue(wz_max_boost, value_type=float),
            "axis_wz_boost_l": ParameterValue(axis_wz_boost_l, value_type=int),
            "axis_wz_boost_r": ParameterValue(axis_wz_boost_r, value_type=int),
            "wz_boost_rest": ParameterValue(wz_boost_rest, value_type=float),
            "wz_boost_on": ParameterValue(wz_boost_on, value_type=float),
            "wz_boost_off": ParameterValue(wz_boost_off, value_type=float),
            # ── 명령 스무딩 (슬루 레이트) ──
            #   스틱을 놓는 순간 목표가 max → 0 으로 떨어지는 계단을
            #   가감속 한계로 완만하게 만든다. 가속은 크게(응답성),
            #   감속은 그보다 작게(부드러운 정지) 잡는다.
            #   E-STOP/데드맨 해제는 노드 내부에서 슬루를 우회하므로
            #   긴급 정지 응답성은 그대로다.
            "slew_enable": ParameterValue(slew_enable, value_type=bool),
            "accel_vx": ParameterValue(accel_vx, value_type=float),
            "decel_vx": ParameterValue(decel_vx, value_type=float),
            "accel_wz": ParameterValue(accel_wz, value_type=float),
            "decel_wz": ParameterValue(decel_wz, value_type=float),
            #   버튼 실측: A=0, B=1, X=3, Y=4
            "button_kick": ParameterValue(button_kick, value_type=int),   # A
            "button_estop": 1,        # B: 즉시 stop 모드
            "button_freq_reset": 4,   # Y: 주파수 기본값 복귀
            "enable_button": ParameterValue(enable_button, value_type=int),

            # ── 보행 모드/주파수 ──
            # stop 모드는 순환에서 뺐다. auto 에서 스틱을 놓으면 정책이 알아서
            # 발을 모으고 멈추므로(v0_2 는 정지 시 위상 0 수렴) stop 은 중복이고,
            # 순환 단계가 하나 줄어 조작이 단순해진다.
            # B 버튼(E-STOP)은 그대로 동작한다 — 목록에 stop 이 없으면
            # 첫 모드(auto)로 되돌리고 속도를 0 으로 만든다.
            "gait_modes": ["auto", "march"],
            "gait_mode_default": gait_mode,
            "gait_freq_default": ParameterValue(gait_freq, value_type=float),
            # ★ 학습 분포와 반드시 일치시킬 것 ★ (2026-08-24 갱신)
            #   교체된 보행 정책은 주파수 1.7~2.4 Hz 로 학습됐으므로
            #   그 범위를 그대로 min/max 에 넣는다. 노드는 이 두 값으로
            #   D-pad 입력을 클램프하므로(joypad_teleop_node._step_freq),
            #   여기가 분포 밖 주파수를 막는 유일한 지점이다.
            #   러너 쪽 cb_gait_frequency 는 경고만 하고 값은 그대로 쓴다.
            #
            #   [과거 이력] v0_2(주파수 2.0 고정) 시절에는 min=max=2.0 으로
            #   잠가 D-pad 상하를 무효화해 뒀다. 정책이 다시 고정 주파수로
            #   돌아가면 같은 방식으로 잠글 것.
            "gait_freq_min": ParameterValue(gait_freq_min, value_type=float),
            "gait_freq_max": ParameterValue(gait_freq_max, value_type=float),
            #   D-pad 1 회당 변화량. 0.1 이면 기본값 2.0 에서
            #   하한 1.7 까지 3 회, 상한 2.4 까지 4 회 입력으로 도달한다.
            #   (한계에 닿으면 노드가 경고 로그를 한 번 남기고 값을 유지)
            "gait_freq_step": ParameterValue(gait_freq_step, value_type=float),
        }],
    )

    # ── 킥 러너는 control.launch.py 로 이관됐다 (2026-08-18) ────────────
    #   예전에는 이 파일에서 actor_onnx_walkkick_v1_runner 를 직접 띄웠다.
    #   그러면 경기용 match.launch.py 에는 킥 노드가 없어서, 비전 서보가
    #   /kick_trigger 를 쏴도 듣는 노드가 없고 FSM 이 KICK_WAIT 에서 멈췄다.
    #   보행 러너와 킥 러너는 /joint_goals 소유권을 주고받는 한 쌍이므로
    #   같은 곳(control.launch.py)에서 함께 관리하고, 여기서는 use_kick 과
    #   튜닝 인자만 넘긴다. 조이패드 A 버튼과 비전 서보가 같은 경로를 쓴다.

    return LaunchDescription([
        DeclareLaunchArgument("robot", default_value=""),
        # ── 보행 정책 / 모드 ──
        DeclareLaunchArgument(
            "policy_variant", default_value="v0_2",
            description="정책 변형. ONNX 관측 차원과 반드시 일치 (v0_1|v0_2|v1|v1s)"),
        DeclareLaunchArgument(
            "gait_mode", default_value="auto",
            description="시작 보행 모드 (auto|march). D-pad 좌우로 이 둘만 순환한다. "
                        "stop 은 auto+속도0 과 중복이라 뺐다"),
        DeclareLaunchArgument(
            "gait_freq", default_value="2.0",
            description="시작 보행 주파수 [Hz]. Y 버튼 복귀값도 이 값이다. "
                        "가변 범위 1.7~2.4 의 안쪽이면서 0.1 스텝으로 "
                        "양쪽 한계에 정확히 닿는 값으로 2.0 을 쓴다. "
                        "이 값은 control.launch.py 로도 전달되어 러너의 "
                        "초기 주파수가 된다(조이패드 latched 발행과 일치)."),
        DeclareLaunchArgument(
            "gait_freq_min", default_value="1.7",
            description="D-pad 하 로 내릴 수 있는 하한 [Hz]. "
                        "교체된 정책의 학습 분포 하한(1.7)과 일치시킨 값. "
                        "학습 분포 밖으로 내리면 성능 보장이 안 된다."),
        DeclareLaunchArgument(
            "gait_freq_max", default_value="2.4",
            description="D-pad 상 으로 올릴 수 있는 상한 [Hz]. "
                        "교체된 정책의 학습 분포 상한(2.4)과 일치시킨 값. "
                        "주파수를 올리면 스텝이 짧고 빨라진다 — 첫 확인은 "
                        "반드시 매단 상태에서."),
        DeclareLaunchArgument(
            "gait_freq_step", default_value="0.1",
            description="D-pad 1 회 입력당 주파수 변화량 [Hz]. "
                        "min/max 를 정확히 밟으려면 (max-min) 이 이 값의 "
                        "정수배가 되게 잡는 것이 깔끔하다."),
        DeclareLaunchArgument("joy_dev", default_value="0"),
        DeclareLaunchArgument(
            "joy_deadzone", default_value="0.05",
            description="joy_node 자체 데드존(raw). 스틱 중립 오차 제거용. "
                        "joy_node 가 재정규화하므로 조작 범위는 줄지 않는다."),
        DeclareLaunchArgument(
            "teleop_deadzone", default_value="0.20",
            description="속도 매핑 데드존(전 축 공통). 스틱을 이 비율 이상 "
                        "밀어야 속도가 나간다. 최소속도(*_min)가 0 이라 "
                        "넉넉히 잡아도 조작감 손해가 없다."),
        DeclareLaunchArgument("enable_button", default_value="-1"),
        DeclareLaunchArgument("button_kick", default_value="0"),
        DeclareLaunchArgument(
            "deadzone_vy", default_value="-1.0",
            description="좌우(vy) 전용 데드존. -1.0 이면 teleop_deadzone 상속. "
                        "전진 중 대각선으로 새면 0.30~0.35 로 키운다."),
        DeclareLaunchArgument(
            "wz_max", default_value="1.5",
            description="오른쪽 스틱 최대 회전속도 [rad/s]. 스틱을 끝까지 "
                        "밀었을 때 나가는 명령값이다. "
                        "⚠ 명령값이지 실제 회전속도가 아니다 — 실측 스케일 "
                        "약 0.7 (명령 1.0 → 실제 약 0.7 rad/s, "
                        "nav_cmd_bridge_node 주석 참조). "
                        "⚠ 학습 분포(명령 wz 범위) 밖으로 올리면 정책이 "
                        "본 적 없는 조건이라 성능 보장이 안 된다. "
                        "올릴 때는 반드시 매단 상태에서 먼저 확인할 것."),
        DeclareLaunchArgument(
            "wz_min", default_value="0.5",
            description="데드존을 넘는 순간 나가는 최소 회전속도 [rad/s]. "
                        "0 이 아니면 0 → wz_min 계단이 생겨 스틱을 놓을 때 "
                        "휘청임의 원인이 된다. 그 증상이 보이면 0.0 으로."),
        DeclareLaunchArgument(
            "wz_max_boost", default_value="3.0",
            description="LT/RT 를 당기고 있는 동안 적용할 회전속도 상한 "
                        "[rad/s]. 놓으면 즉시 wz_max 로 복귀한다. "
                        "0.0 이면 부스트 기능 자체를 끈다. "
                        "⚠ 학습 분포 밖일 가능성이 크다 — 반드시 매단 "
                        "상태에서 2.0 → 2.5 → 3.0 순으로 올려 확인할 것."),
        DeclareLaunchArgument(
            "axis_wz_boost_l", default_value="4",
            description="LT 트리거의 axes 인덱스. -1 이면 LT 미사용. "
                        "★ 기본값 4/5 는 hid-generic 계열에서 흔한 값일 "
                        "뿐 보장값이 아니다. `ros2 topic echo /joy` 로 "
                        "반드시 실측할 것."),
        DeclareLaunchArgument(
            "axis_wz_boost_r", default_value="5",
            description="RT 트리거의 axes 인덱스. -1 이면 RT 미사용. "
                        "LT/RT 는 OR 조건이라 아무 쪽이나 당기면 걸린다."),
        DeclareLaunchArgument(
            "wz_boost_rest", default_value="1.0",
            description="트리거를 놓았을 때의 축 값. +1.0(놓으면 +1, "
                        "당기면 -1) 또는 -1.0(그 반대). echo /joy 에서 "
                        "트리거를 놓고 본 값을 그대로 넣으면 된다. "
                        "부호가 틀리면 놓은 상태가 '완전히 당김' 으로 "
                        "해석되지만, 무장 래치가 막아 주므로 부스트가 "
                        "안 걸리는 증상으로 나타난다."),
        DeclareLaunchArgument(
            "wz_boost_on", default_value="0.60",
            description="부스트 ON 임계 (트리거를 당긴 정도 0~1). "
                        "낮출수록 살짝만 당겨도 걸린다."),
        DeclareLaunchArgument(
            "wz_boost_off", default_value="0.40",
            description="부스트 OFF 임계. wz_boost_on 보다 반드시 작아야 "
                        "한다. 두 값의 간격이 히스테리시스 폭이고, 이게 "
                        "없으면 임계 근처에서 상한이 깜빡이며 회전 명령이 "
                        "진동한다."),
        DeclareLaunchArgument(
            "slew_enable", default_value="true",
            description="명령 스무딩 on/off. false 면 예전처럼 계단 그대로."),
        DeclareLaunchArgument(
            "accel_vx", default_value="3.0",
            description="vx 가속 한계 [m/s^2]. 클수록 출발이 즉각적이다."),
        DeclareLaunchArgument(
            "decel_vx", default_value="2.0",
            description="vx 감속 한계 [m/s^2]. 작을수록 정지가 부드럽지만 "
                        "정지 거리가 길어진다. 1.0 미만은 권하지 않는다."),
        DeclareLaunchArgument(
            "accel_wz", default_value="6.0",
            description="wz 가속 한계 [rad/s^2]."),
        DeclareLaunchArgument(
            "decel_wz", default_value="4.0",
            description="wz 감속 한계 [rad/s^2]. wz_min=0.5 계단이 남아 "
                        "있으므로 vx 보다 이쪽 효과가 크다."),
        DeclareLaunchArgument(
            "walk_actor_path",
            # 2026-08 정책 교체: policy.onnx(구 walk) → walk_policy.onnx(v1s 러너)
            # ★ 이 인자는 control.launch.py 로 '명시 전달' 되므로, 여기 기본값이
            #   control.launch.py 의 기본값을 덮어쓴다. 모델 파일명을 바꿀 때는
            #   control.launch.py 와 이 파일을 반드시 같이 고칠 것.
            default_value=os.path.join(pol_models, "walk_policy.onnx")),
        DeclareLaunchArgument(
            "residual_actor_path",
            default_value=os.path.join(pol_models, "residual_kick_power_policy.onnx")),
        DeclareLaunchArgument("reference_csv_path", default_value=""),
        # ── 킥(walkkick_v1) 인자 ──
        DeclareLaunchArgument(
            "use_kick", default_value="true",
            description="A 버튼 킥 러너 실행 여부. false 면 보행만."),
        DeclareLaunchArgument(
            "kick_ball_x", default_value="0.25",
            description="공까지 전방 거리 [m] (base_link 기준 고정값). "
                        "학습 스폰 범위 0.20~0.50 m. 0.20 은 하한 경계라 "
                        "여유를 두어 0.25 를 기본값으로 한다."),
        DeclareLaunchArgument(
            "kick_cmd_vx", default_value="0.3",
            description="킥 중 접근 전진 명령 [m/s]. /cmd_vel 구독을 끄므로 "
                        "이 값이 그대로 유지된다. 학습 분포 0.1~0.5 의 중앙이 "
                        "0.3 이다. ⚠ 하한 0.1 은 gait standing_threshold(0.1)와 "
                        "같아, 0.15 미만이면 명령 노름이 임계에 걸려 freq=0 이 "
                        "되고 킥 중에 걷지 않는다."),
        DeclareLaunchArgument(
            "kick_duration", default_value="2.0",
            description="트리거 후 킥 구동 시간 [s]. 이 시간이 지나면 "
                        "/kick_done 을 내고 보행에 소유권을 돌려준다."),
        DeclareLaunchArgument(
            "kick_start_hold", default_value="0.3",
            description="[킥 전] 트리거 직후 default 자세 hold [s]. "
                        "킥 정책 급발진 방지. 킥 후 hold 는 kick_end_hold."),
        DeclareLaunchArgument(
            "kick_strict_gate", default_value="true",
            description="학습 분포 게이트. kick_ball_x 가 0.20~0.50 m 밖이면 "
                        "false 여야 트리거가 통과한다. 기본값 0.25 는 분포 "
                        "안이므로 게이트를 켜둔다(OOD 발산 방지)."),
        DeclareLaunchArgument(
            "kick_end_hold", default_value="0.0",
            description="[킥 후] 킥 종료 → 보행 복귀 hold [s]. "
                        "-1.0=기본(1.0s) 상속, 0.0=hold 없이 즉시 보행. "
                        "★0.0 은 last_action 이 0 으로 리셋된 채 첫 추론이 "
                        "나가므로 발산 위험이 있다. 반드시 매단 상태에서 "
                        "먼저 확인할 것. 킥 전 hold 는 kick_start_hold."),
        control,
        joy_node,
        joypad_node,
    ])
