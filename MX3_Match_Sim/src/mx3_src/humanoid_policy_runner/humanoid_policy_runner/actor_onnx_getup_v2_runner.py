#!/usr/bin/env python3
# actor_onnx_getup_v2_runner.py
#
# ============================================================================
#  supine get-up V2 (보행 통합형 기상) 정책 실기 테스트 러너
# ============================================================================
#  학습 환경 : Unitree-MX3-Getup-V2
#              unitree_rl_lab/tasks/locomotion/robots/MX3/getup_v2_env_cfg.py
#  로봇 자산 : mx_bot.ROBOT_CONFIGS["XM_robot04_getup"]  (팔 구동 ON, 18 DOF)
#
#  ★ v1 러너(actor_onnx_getup_runner.py)와 모델 호환이 없다 ★
#    관측 차원이 300 -> 330 으로 바뀌었다. v1 모델을 이 노드에,
#    v2 모델을 v1 노드에 물리면 ONNX 로드 직후 차원 검사에서 죽는다.
#    (죽지 않고 통과하는 경우는 없다 — 300 != 330 이라 반드시 걸린다)
#
# ============================================================================
#  【v1 -> v2 에서 무엇이 바뀌었나】 — 이 파일이 존재하는 이유
# ============================================================================
#  getup_v2_env_cfg.py 헤더 요약: v2 는 "기상 전용 env" 가 아니라
#  **보행 env(velocity_v0_6)를 베이스로 기상을 얹은** 통합 환경이다.
#  v1 이 반복 실패한 원인이 전부 "게이트 조작"(로봇이 게이트 문턱 바로 위에
#  머물러 페널티를 회피)이었기 때문에, v2 는 게이팅을 쓰지 않고 역방향
#  커리큘럼이 그 역할을 대신한다.
#
#  그 결과 **관측에 보행 명령 두 개가 추가**됐다:
#
#     term                 v1        v2         비고
#     ───────────────────  ────      ────       ───────────────────────────
#     base_ang_vel          3         3
#     projected_gravity     3         3
#     velocity_commands     -         3   ★신규  (vx, vy, wz)
#     gait_phase            -         3   ★신규  (cos, sin, freq/scale)
#     joint_pos_rel        18        18
#     joint_vel_rel        18        18
#     last_action          18        18
#     ───────────────────  ────      ────
#     프레임               60        66
#     x history 5         300       330
#
#  액션은 그대로 18 (다리 12 + 팔 6), action_scale 0.25, use_default_offset.
#  액션 원점(default_joint_pos)도 v1 과 같다 — 같은 로봇 자산을 쓴다.
#
#  ※ 차원 근거: MX3/__init__.py 의 Getup-V2 등록 주석에
#    "검사2 대합 PASS (policy 330 / critic 345 / actions 18, 오차 0)"
#    로 남아 있다. critic 345 = 여기에 base_lin_vel(3) 이 더해진 값이다.
#
# ============================================================================
#  관측 규격 (학습 ObservationsCfg.PolicyCfg 선언 순서 그대로)
# ============================================================================
#       [0:3]     base_ang_vel        * 0.2
#       [3:6]     projected_gravity
#       [6:9]     velocity_commands   (vx, vy, wz)          ← v2 신규
#       [9:12]    gait_phase          (cos, sin, freq/2.0)  ← v2 신규
#       [12:30]   joint_pos_rel       (18)
#       [30:48]   joint_vel_rel       * 0.05  (18)
#       [48:66]   last_action         (18)
#
#     프레임 66, history_length 5  →  관측 330
#
#  ★ history concat 은 term-major ★
#     Isaac Lab 은 각 term 의 5 프레임을 먼저 이어붙이고 다음 term 으로
#     넘어간다. 프레임을 통째로 5 번 붙이는(frame-major) 방식이 아니다.
#     symmetry_mx3._build_index_map 이 term 단위로 history 를 반복하며
#     인덱스를 만드는 것이 그 증거다. 순서를 틀리면 정책은 전혀 다른 값을
#     보게 되는데 **에러는 나지 않는다** — 조용히 오동작한다.
#
# ============================================================================
#  【신규 관측 1】 velocity_commands — 실기에서 무엇을 넣어야 하나
# ============================================================================
#  학습 RobotPlayEnvCfg (= 개발자가 의도한 추론 조건) 은 이렇다:
#      ranges            = 전부 (0.0, 0.0)
#      rel_standing_envs = 1.0          → 전 env 속도 0
#  즉 **추론 기준 조건은 속도 명령 0** 이다.
#
#  그래서 이 노드의 기본값은 cmd_vel_source="zero" 다. 기상만 시험한다.
#
#  다만 v2 는 보행까지 같은 정책이 하므로, 일어선 뒤 그대로 걷게 할 수 있다.
#  그 시험이 필요하면 cmd_vel_source="subscribe" 로 /cmd_vel 을 받는다.
#    ※ 안전장치: cmd_vel_require_upright_deg (기본 25도) 보다 많이 기울어져
#      있으면 속도 명령을 0 으로 강제한다. 누워 있는 동안 전진 명령이
#      들어가면 학습 Play 조건과 어긋난다.
#    ※ 학습 limit_ranges 밖 명령은 분포 밖이다. cmd_vel_limits 로 자른다:
#        lin_x (-0.5, 0.7) / lin_y (-0.5, 0.5) / ang_z (-1.5, 1.5)
#
# ============================================================================
#  【신규 관측 2】 gait_phase — 실기에서 위상을 직접 만들어야 한다
# ============================================================================
#  학습의 PhaseOnlyGaitCommand (htwk_gait_v01.py) 를 그대로 이식했다.
#  그 파일 헤더의 "실기 배포 시 반드시 동일하게 구현할 것" 세 줄이 요구사항이다:
#     - 증분 적분 phase = fmod(phase + dt*freq, 1.0)
#       (절대 시각 기반으로 만들면 주파수를 바꾸는 순간 위상이 점프한다)
#     - 정지 시 위상을 0 으로 즉시 리셋하지 말 것. 최단 경로로 수렴시킬 것
#     - cos/sin 게이팅 없음
#
#  관측 3 차원 = (cos 2πφ, sin 2πφ, freq / frequency_scale).
#  셋째 성분이 "제자리 걷기(freq 유지)" 와 "완전 정지(freq 0)" 를 가른다.
#
#  ── gait_mode 로 무엇을 골라야 하나 ──
#    학습 Play 조건에서 전 env 가 속도 0 이고, 그중
#      rel_stop_when_standing = 0.6  →  60 % 완전 정지 / 40 % 제자리 걷기
#    두 모드 모두 분포 안이다. 이 노드의 기본값은 "stop" 이다:
#      - 관측이 (1, 0, 0) 상수라 가장 조용하다. 기상 동작만 분리해서 본다.
#      - 학습 비중도 이쪽이 더 크다 (60 %).
#    일어선 뒤 리듬을 유지시키려면 gait_mode:="march" 로 바꾼다.
#      ros2 topic pub --once /getup_v2/gait_mode std_msgs/String "{data: march}"
#
#  ── 주파수 ──
#    학습은 frequency_range=(1.7, 2.4) 로 env 마다 랜덤이다. 실기에서는
#    하나를 골라야 하므로 중앙값에 가까운 2.0 을 기본값으로 쓴다.
#    범위 밖 값을 주면 분포 밖이므로 경고를 띄운다.
#
# ============================================================================
#  시퀀스 (v1 과 동일 구조. 검증된 절차라 그대로 유지한다)
# ============================================================================
#       [IDLE]  트리거 대기 (/joint_goals 발행 안 함)
#          │  ← /getup_trigger(Bool) 또는 서비스 ~/start
#          ▼
#       [PREP]   prep_duration_s (기본 2.0 s)
#                현재 자세 → 학습 기본자세(액션 원점) 로 코사인 보간.
#                실기에서 뒤로 넘어지면 팔꿈치는 접힌 채(약 3.04 rad)로
#                굳는데, 학습 액션 원점은 팔꿈치 ±1.5 다. 이 구간이 그
#                차이를 메운다. 이걸 건너뛰면 정책이 낼 수 있는 목표
#                범위 전체가 1.5 rad 통째로 밀린다.
#
#                ★ 이 구간은 학습의 reset_hold(0.1 s) 에 대응한다 ★
#                  학습 hold 동안:
#                    - generated_commands_with_reset_hold → 속도 명령 0
#                    - freeze_during_reset_hold → gait 주파수 0
#                  그래서 PREP 동안 이 노드도 속도 0 / 주파수 0 을 쓴다.
#          ▼
#       [POLICY] policy_duration_s (기본 10.0 s)  ← **안전장치**
#                ONNX 추론 → /joint_goals. 시간이 다 되면 무조건 종료한다.
#                0 이하면 무제한(직접 abort 해야 한다. 권장하지 않는다).
#          ▼
#       [FINISH] end_action 에 따라 마무리 (hold / walk_ready / none)
#
# ============================================================================
#  실기에서 반드시 챙겨야 하는 것 — v1 과 동일
# ============================================================================
#  (1) 팔 Goal PWM 상한
#      motor_gains.yaml 의 arms 그룹에 goal_pwm: 50 (885=100% 이므로 약 5.6%)
#      이 걸려 있다. 실측상 1.0 rad 이동에 최소 150 이 필요하므로 **이대로면
#      팔이 아예 안 움직인다.** 이 노드가 시작 시 /u2d2/set_joint_goal_pwm
#      으로 팔 6 개를 100% 로 올리고, 끝나면 reload_gains 로 되돌린다.
#
#  (2) 팔 PID 게인
#      arms 는 P=800, D=0 이다(자세 유지만 하던 시절 값). 다리는 P=800,
#      D=500. 팔로 체중을 밀 때 D=0 이면 진동할 수 있다. 이 노드는 게인을
#      건드리지 않는다 — 보행에도 영향이 가는 값이라 자동 변경하지 않는다.
#      떨리면 motor_gains.yaml 의 arms 에 position_d_gain 을 넣고
#      reload_gains 로 반영할 것.
#
#  (3) Profile Velocity
#      robot_init 이 끝나면 45 rpm(= 4.71 rad/s)이다. sim 실측 기상 관절속도
#      4.57 rad/s 라 거의 한계선이다. 여유가 필요하면
#      policy_profile_velocity_rpm 을 50 정도로 (기본값 -1 = 미변경).
#      ※ XM540-W150 은 11.1 V 무부하 50 rpm 이라 그 이상은 의미가 없다.
#      ※ 0 은 다이나믹셀에서 "무제한" 특수값이다. 절대 넣지 말 것.
#
#  (4) fall_manager_node 를 **반드시 끄고** 테스트할 것
#      로봇이 누워 있으면 FALLEN 으로 판정해 standup 키프레임을 재생하고
#      /joint_goals 에 보호자세를 직접 쏜다. 이 노드와 매 틱 서로를 덮어써서
#      관절이 진동한다. 이 노드는 /robot_state 를 구독해 경고만 띄운다
#      (막지는 않는다 — 막으면 누운 상태에서 영영 못 돈다).
#
# ============================================================================
#  실행
# ============================================================================
#    # 1) 하드웨어 스택 (fall_manager 없이)
#    ros2 launch humanoid_serial_bridge dynamixel_system.launch.py
#
#    # 2) 이 노드
#    ros2 launch humanoid_policy_runner getup_v2_test.launch.py \
#        actor_path:=/home/robo02/models/getup_v2.onnx
#
#    # 첫 시험은 dry_run 권장 — 관절을 안 움직이고 관측만 CSV 로 남긴다.
#    ros2 launch humanoid_policy_runner getup_v2_test.launch.py \
#        actor_path:=... dry_run:=true
#
#    # 3) 로봇을 등 대고 눕힌 뒤 트리거
#    ros2 service call /actor_onnx_getup_v2_node/start std_srvs/srv/Trigger
#      또는
#    ros2 topic pub --once /getup_trigger std_msgs/Bool "{data: true}"
#
#    # 중단 (언제든)
#    ros2 service call /actor_onnx_getup_v2_node/abort std_srvs/srv/Trigger
#
#    # /joint_goals 점유 해제 (보행 러너로 넘기기 전)
#    ros2 service call /actor_onnx_getup_v2_node/reset std_srvs/srv/Trigger
#
#  CSV: ~/policy_logs/getup_v2_log_YYYYMMDD_HHMMSS.csv
# ============================================================================

import csv
import math
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Point, Twist, Vector3Stamped
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

# projected gravity 파생 계산. 보행/기상 v1 러너와 같은 헬퍼를 쓴다.
#   /imu/projected_gravity 를 따로 구독하지 않고 /imu/data 의 orientation 에서
#   직접 만든다(값은 imu_parser 가 보내주던 것과 비트 단위로 동일하다).
from humanoid_serial_bridge.imu_math import projected_gravity_from_quat
from humanoid_behavior_interfaces.srv import TorqueEnable

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError(
        "onnxruntime이 설치되어 있지 않습니다.\n"
        "  CPU:  pip install onnxruntime\n"
        "  GPU:  pip install onnxruntime-gpu"
    )

try:
    from ament_index_python.packages import get_package_share_directory
except Exception:                                               # noqa: BLE001
    get_package_share_directory = None


# ============================================================================
#  학습 환경에서 그대로 옮겨온 상수들
#  ※ 여기 값이 학습과 어긋나면 정책은 조용히 오동작한다(에러가 안 난다).
#    getup_v2_env_cfg.py / htwk_gait_v01.py / mx_bot.py 를 고치면 반드시
#    이쪽도 같이 고칠 것.
# ============================================================================

# getup_v2_env_cfg.GETUP_JOINTS = MX_LEG_JOINTS + MX_ARM_JOINTS 와 같은 순서.
#   ActionsCfg / ObservationsCfg 가 preserve_order=True 로 이 순서를
#   고정하므로, 한 칸이라도 어긋나면 관측·액션이 통째로 뒤섞인다.
GETUP_JOINTS = [
    # ── 다리 12 ──
    "r_pelvis_pitch_joint",
    "r_pelvis_roll_joint",
    "r_pelvis_yaw_joint",
    "r_knee_pitch_joint",
    "r_ankle_pitch_joint",
    "r_ankle_roll_joint",
    "l_pelvis_pitch_joint",
    "l_pelvis_roll_joint",
    "l_pelvis_yaw_joint",
    "l_knee_pitch_joint",
    "l_ankle_pitch_joint",
    "l_ankle_roll_joint",
    # ── 팔 6 ──
    "r_shoulder_pitch_joint",
    "r_shoulder_roll_joint",
    "r_elbow_pitch_joint",
    "l_shoulder_pitch_joint",
    "l_shoulder_roll_joint",
    "l_elbow_pitch_joint",
]
GETUP_JOINTS_SET = set(GETUP_JOINTS)
NUM_JOINTS = len(GETUP_JOINTS)          # 18
HISTORY_LEN = 5

# ── 액션 원점 = 학습 default_joint_pos ──
#   mx_bot.ROBOT_CONFIGS["XM_robot04_getup"].init_joint_pos 그대로.
#   ★ v1 과 완전히 같다 ★ v2 도 같은 로봇 자산(XM_robot04_getup)을 쓴다.
#   ★ 그 dict 에 없는 관절(어깨 pitch/roll, 골반 roll/yaw, 발목 roll)은
#     Isaac Lab 에서 0.0 이 된다. 여기서도 0.0 으로 명시해 둔다.
#   ★ 팔꿈치가 보행(±2.967)과 다른 ±1.5 인 것이 핵심이다.
#     보행 기본각은 팔꿈치 리밋(0, 2.967)의 끝단이라 action*scale 로는
#     펼 수가 없어서, 학습 때 리밋 중점으로 옮겨 놓았다.
GETUP_DEFAULT_Q = {
    "r_pelvis_pitch_joint": 0.62,
    "r_pelvis_roll_joint": 0.0,
    "r_pelvis_yaw_joint": 0.0,
    "r_knee_pitch_joint": 1.2,
    "r_ankle_pitch_joint": 0.62,
    "r_ankle_roll_joint": 0.0,

    "l_pelvis_pitch_joint": -0.62,
    "l_pelvis_roll_joint": 0.0,
    "l_pelvis_yaw_joint": 0.0,
    "l_knee_pitch_joint": -1.2,
    "l_ankle_pitch_joint": -0.62,
    "l_ankle_roll_joint": 0.0,

    "r_shoulder_pitch_joint": 0.0,
    "r_shoulder_roll_joint": 0.0,
    "r_elbow_pitch_joint": 1.5,       # ← 보행은 2.967
    "l_shoulder_pitch_joint": 0.0,
    "l_shoulder_roll_joint": 0.0,
    "l_elbow_pitch_joint": -1.5,      # ← 보행은 -2.967
}

# ── 관절 목표 클램프 ──
#   ★ v1 과 v2 는 클램프 소스가 다르지만 값은 같다 ★
#     v1 : GetupJointPositionActionCfg(target_limits=_TARGET_LIMITS)  명시 dict
#     v2 : GetupJointPositionActionCfg(target_limits=None)            USD soft limit
#   reset_hold.DelayedJointPositionAction._build_clamp_limits 를 보면
#   target_limits=None 일 때 asset.data.soft_joint_pos_limits 를 쓴다.
#   그런데 v1 의 _TARGET_LIMITS 자체가 그 USD 실측값이고(주석: "soft limit ==
#   hard limit 이라 그대로 target 클램프로 쓴다"), 두 태스크가 같은 자산
#   (XM_robot04_getup)을 쓰므로 결과가 동일하다.
#   → 같은 표를 그대로 쓴다. USD 를 교체하면 여기도 다시 떠야 한다.
TARGET_LIMITS = {
    "r_pelvis_pitch_joint": (-1.5708, +1.5708),
    "r_pelvis_roll_joint": (-0.4363, +1.3963),
    "r_pelvis_yaw_joint": (-1.5708, +1.5708),
    "r_knee_pitch_joint": (+0.0000, +2.4435),
    "r_ankle_pitch_joint": (-0.8727, +1.8326),
    "r_ankle_roll_joint": (-0.5236, +0.5236),
    "l_pelvis_pitch_joint": (-1.5708, +1.5708),
    "l_pelvis_roll_joint": (-1.3963, +0.4363),
    "l_pelvis_yaw_joint": (-1.5708, +1.5708),
    "l_knee_pitch_joint": (-2.4435, +0.0000),
    "l_ankle_pitch_joint": (-1.8326, +0.8727),
    "l_ankle_roll_joint": (-0.5236, +0.5236),
    "r_shoulder_pitch_joint": (-2.6180, +2.6180),
    "r_shoulder_roll_joint": (-0.2618, +2.8798),
    "r_elbow_pitch_joint": (+0.0000, +2.9671),
    "l_shoulder_pitch_joint": (-2.6180, +2.6180),
    "l_shoulder_roll_joint": (-2.8798, +0.2618),
    "l_elbow_pitch_joint": (-2.9671, +0.0000),
}

# 팔 관절만 따로. Goal PWM 승격 대상이다(다리는 원래 상한이 없다).
ARM_JOINTS = [j for j in GETUP_JOINTS if ("shoulder" in j or "elbow" in j)]

# 학습 frequency_range=(1.7, 2.4). 이 밖은 분포 외라 경고만 띄운다.
GAIT_FREQ_TRAINED_RANGE = (1.7, 2.4)

# 학습 limit_ranges (커리큘럼이 최종적으로 확장하는 범위).
#   getup_v2_env_cfg.CommandsCfg.base_velocity.limit_ranges 와 같은 값이다.
#   [x_lo, x_hi, y_lo, y_hi, wz_lo, wz_hi]
CMD_VEL_LIMITS_DEFAULT = [-0.5, 0.7, -0.5, 0.5, -1.5, 1.5]

# ── 상태 문자열 ──
ST_WAIT_READY = "WAIT_READY"    # 센서(/joint_states, /imu/data) 대기
ST_IDLE = "IDLE"                # 트리거 대기. /joint_goals 발행 안 함
ST_PREP = "PREP"                # 초기자세 만들기 (= 학습 reset_hold 대응)
ST_POLICY = "POLICY"            # 정책 추론
ST_FINISH = "FINISH"            # 마무리 (hold / walk_ready 전이)
ST_ABORT = "ABORT"              # 중단 — 현재 자세로 굳힘

GAIT_MODES = ("stop", "march", "walk", "auto")


def _expand_path(p: str) -> str:
    return str(Path(os.path.expanduser(p)).resolve())


def _cos_ease(t: float) -> float:
    """0→1 코사인 이징. 양 끝에서 속도가 0 이라 관절이 튀지 않는다.

    선형 보간을 쓰면 시작·끝 순간에 속도가 계단으로 바뀌어 팔이 튄다.
    특히 PREP 구간은 로봇이 누운 채 팔꿈치를 1.5 rad 이나 움직이므로
    시작 충격이 그대로 몸통을 흔든다.
    """
    t = min(1.0, max(0.0, float(t)))
    return 0.5 * (1.0 - math.cos(math.pi * t))


def _load_walk_ready_pose() -> dict:
    """end_action="walk_ready" 에서 쓸 '보행 시작 자세'.

    ★ 학습 기본각이 아니라 **실기 robot_init 자세**를 쓴다 ★
      v2 는 보행까지 같은 정책이 하므로 팔꿈치 ±1.5 를 유지해도 되지만,
      이 시험을 끝내고 다른 러너(보행 v0_2/v1s)로 넘길 수 있어야 한다.
      그쪽은 팔을 robot_init_behavior 의 arm_init_q(어깨 pitch ±0.2,
      팔꿈치 ±3.04)로 잡고 있으므로 그 자세로 되돌려 두는 편이 안전하다.

    import 에 실패하면(패키지 구조 변경 등) 하드코딩 값으로 폴백한다.
    """
    try:
        from humanoid_serial_bridge.behaviors.robot_init_behavior import (
            RobotInitBehavior,
        )
        rib = RobotInitBehavior(hw=None)
        pose = {}
        pose.update({k: float(v) for k, v in rib.leg_init_q.items()})
        pose.update({k: float(v) for k, v in rib.arm_init_q.items()})
        return pose
    except Exception:                                           # noqa: BLE001
        pass

    return {
        "r_pelvis_pitch_joint": 0.62,
        "r_pelvis_roll_joint": 0.0,
        "r_pelvis_yaw_joint": 0.0,
        "r_knee_pitch_joint": 1.2,
        "r_ankle_pitch_joint": 0.62,
        "r_ankle_roll_joint": 0.0,
        "l_pelvis_pitch_joint": -0.62,
        "l_pelvis_roll_joint": 0.0,
        "l_pelvis_yaw_joint": 0.0,
        "l_knee_pitch_joint": -1.2,
        "l_ankle_pitch_joint": -0.62,
        "l_ankle_roll_joint": 0.0,
        "r_shoulder_pitch_joint": 0.2,
        "r_shoulder_roll_joint": 0.0,
        "r_elbow_pitch_joint": 3.04,
        "l_shoulder_pitch_joint": -0.2,
        "l_shoulder_roll_joint": 0.0,
        "l_elbow_pitch_joint": -3.04,
    }


class FrameHistory:
    """관측 프레임 링버퍼. 보행 러너 / getup v1 러너와 동일한 구현이다.

    reset() 은 같은 프레임으로 5 칸을 전부 채운다. 학습에서도 에피소드
    리셋 직후 히스토리 버퍼가 첫 관측으로 가득 차므로 규약이 같다.
    """

    def __init__(self, H: int, dim: int):
        self.H = H
        self.dim = dim
        self.deq = deque(maxlen=H)

    def reset(self, frame: np.ndarray) -> np.ndarray:
        self.deq.clear()
        for _ in range(self.H):
            self.deq.append(frame.copy())
        return self.get_frames()

    def push(self, frame: np.ndarray) -> np.ndarray:
        self.deq.append(frame.copy())
        while len(self.deq) < self.H:
            self.deq.appendleft(frame.copy())
        return self.get_frames()

    def get_frames(self) -> np.ndarray:
        return np.stack(list(self.deq), axis=0).astype(np.float32)


def make_frame_spec():
    """프레임 안의 term 별 슬라이스와 총 차원(66).

    학습 getup_v2_env_cfg.ObservationsCfg.PolicyCfg 의 선언 순서 그대로:
        base_ang_vel(3) → projected_gravity(3) → velocity_commands(3)
        → gait_phase(3) → joint_pos_rel(18) → joint_vel_rel(18)
        → last_action(18)

    ★ 선언 순서가 곧 관측 순서다 ★ configclass 는 클래스 본문의 속성
      정의 순서를 그대로 보존하고, ObservationManager 가 그 순서로
      concatenate 한다. env_cfg 에서 term 위치를 옮기면 여기도 옮겨야 한다.
    """
    slices = []
    ofs = 0
    for n in (3, 3, 3, 3, NUM_JOINTS, NUM_JOINTS, NUM_JOINTS):
        slices.append((ofs, ofs + n))
        ofs += n
    return slices, ofs


def build_obs_from_frames(frames: np.ndarray, slices, obs_dim: int) -> np.ndarray:
    """프레임 히스토리 → 1 차원 관측 벡터 (term-major concat).

    Isaac Lab 의 ObsGroup(history_length=5, concatenate_terms=True) 은
    각 term 의 5 프레임을 먼저 나열하고 다음 term 으로 넘어간다.
    프레임을 통째로 5 번 이어붙이는(frame-major) 방식이 아니므로,
    순서를 틀리면 정책이 전혀 다른 값을 보게 된다(에러는 안 난다).
    """
    h, _dim = frames.shape
    assert h == HISTORY_LEN, frames.shape

    parts = []
    for lo, hi in slices:
        parts.append(frames[:, lo:hi].reshape(h * (hi - lo)))

    obs = np.concatenate(parts, axis=0).astype(np.float32)
    assert obs.shape == (obs_dim,), obs.shape
    return obs


class PhaseOnlyGaitIntegrator:
    """학습 htwk_gait_v01.PhaseOnlyGaitCommand 의 실기 이식본.

    ★ 학습 _update_command() 와 식이 한 줄이라도 달라지면 sim-to-real
      갭이 생긴다. 수정할 때는 반드시 양쪽을 함께 고칠 것. ★

    getup_v2_env_cfg.CommandsCfg.gait 설정:
        frequency_range      = (1.7, 2.4)   env 별 랜덤 (실기는 하나를 고른다)
        stop_rate            = 2.0          위상 0 수렴 속도 [phase/s]
        include_frequency    = True         → 관측 3 차원
        frequency_scale      = 2.0
        rel_stop_when_standing = 0.6        정지 env 중 60 % 완전정지
        reset_phase_on_resample  = True
        freeze_during_reset_hold = True

    관측:
        [0] cos(2πφ)
        [1] sin(2πφ)
        [2] freq / frequency_scale     ← "제자리걷기 vs 완전정지" 구분 신호

    게이팅은 없다. 완전정지는 (1, 0, 0) 이 되고, 그 상태가 5 프레임
    히스토리 내내 얼어붙어 있는 것으로 정책이 정지를 인식한다.
    """

    def __init__(self, stop_rate: float = 2.0, eps: float = 1.0e-6):
        self.phase = 0.0                    # 0~1
        self.stop_rate = float(stop_rate)   # 위상 0 수렴 속도 [phase/s]
        self.eps = float(eps)

        self.effective_frequency = 0.0      # 이번 스텝에 실제 적용된 주파수
        self.is_walking = False
        self.is_stopping = False            # 위상이 0 으로 수렴 중

    def reset(self):
        """위상을 0 에서 다시 시작.

        학습 reset_phase_on_resample=True 와 같다. 에피소드 리셋마다
        위상 0(양발 stance 구간)에서 출발하므로, 실기에서도 시퀀스를
        새로 시작할 때는 반드시 이걸 부른다.
        """
        self.phase = 0.0
        self.effective_frequency = 0.0
        self.is_walking = False
        self.is_stopping = False

    def update(self, dt: float, freq_cmd: float):
        """한 스텝 진행.

        Args:
            dt: 정책 주기 [s]. 학습의 env.step_dt (= decimation * sim.dt
                = 4 * 0.005 = 0.02 s) 에 해당한다.
            freq_cmd: 이번 스텝의 주파수 명령 [Hz]. 0 이면 정지.
        """
        freq = max(0.0, float(freq_cmd))
        self.effective_frequency = freq
        self.is_walking = freq > self.eps

        if self.is_walking:
            # ── 보행: 정방향 증분 적분 ──
            #   절대 시각(fmod(now*freq, 1)) 으로 만들면 주파수를 바꾸는
            #   순간 위상이 점프한다. 반드시 증분이어야 한다.
            self.phase = math.fmod(self.phase + dt * freq, 1.0)
            if self.phase < 0.0:
                self.phase += 1.0
            self.is_stopping = False
            return

        # ── 정지: 0 을 향해 최단 경로로 선형 이동 ──
        #   즉시 0 으로 리셋하면 안 된다. 정책은 "위상이 감속해서 멎는"
        #   과도 구간을 보고 감속 동작을 배웠다.
        step = self.stop_rate * dt
        if self.phase > 0.5:
            # 정방향이 가까움. 1.0 에 도달하면 0 으로 랩.
            d = min(1.0 - self.phase, step)
            self.phase = math.fmod(self.phase + d, 1.0)
            if self.phase < self.eps:
                self.phase = 0.0
        else:
            # 역방향이 가까움.
            d = min(self.phase, step)
            self.phase = self.phase - d
            if self.phase < self.eps:
                self.phase = 0.0

        self.is_stopping = self.phase > self.eps

    def observation(self, frequency_scale: float) -> np.ndarray:
        """관측 3 성분 (cos, sin, freq/scale). 게이팅 없음."""
        two_pi = 2.0 * math.pi
        return np.array([
            math.cos(two_pi * self.phase),
            math.sin(two_pi * self.phase),
            self.effective_frequency / max(1e-6, frequency_scale),
        ], dtype=np.float32)

    def state_str(self, standing: bool) -> str:
        """디버그 토픽용 상태 문자열."""
        if self.is_walking:
            return "MARCHING" if standing else "WALKING"
        if self.is_stopping:
            return "STOPPING"
        return "STOPPED"


class ActorOnnxGetupV2Node(Node):
    """기상 V2(보행 통합형) 정책 실기 테스트 노드."""

    def __init__(self):
        super().__init__("actor_onnx_getup_v2_node")

        # ── 기본 모델 경로 ──
        #   패키지 안 models/getup_v2_policy.onnx 를 기본으로 쓴다.
        #   --symlink-install 로 빌드하면 share/humanoid_policy_runner/models
        #   가 src 를 가리키는 심볼릭 링크가 되므로, **모델 파일만 교체할 때는
        #   재빌드가 필요 없다**. 다만 models/ 에 새 파일을 추가한 직후에는
        #   setup.py 의 glob('models/*') 가 빌드 시점에 평가되므로 한 번은
        #   colcon build 를 돌려야 share 에 등록된다.
        default_actor = "~/getup_v2_policy.onnx"
        if get_package_share_directory is not None:
            try:
                default_actor = os.path.join(
                    get_package_share_directory("humanoid_policy_runner"),
                    "models",
                    "getup_v2_policy.onnx",
                )
            except Exception:                                   # noqa: BLE001
                pass

        # ════════════════════════════════════════════════════════════
        #  파라미터 선언
        # ════════════════════════════════════════════════════════════
        self.declare_parameter("actor_path", default_actor)
        self.declare_parameter("onnx_provider", "auto")
        # 정책 주기 [Hz]. 실제 tick 은 /joint_states(50Hz) 이벤트로 돈다.
        # ★ 이 값은 표시용이 아니라 **gait 위상 적분 dt** 로 쓰인다 ★
        #   학습 env.step_dt = decimation(4) * sim.dt(0.005) = 0.02 s = 50 Hz.
        #   실제 tick 이 50 Hz 가 아니면 위상이 학습과 다른 속도로 돈다.
        #   그래서 아래 _check_tick_rate() 가 실측 주기를 감시해 경고한다.
        self.declare_parameter("policy_rate_hz", 50.0)

        # ── 시퀀스 타이밍 ──
        self.declare_parameter("prep_duration_s", 2.0)
        self.declare_parameter("prep_settle_s", 0.0)
        # ★ 안전장치 ★ 이 시간이 지나면 성공/실패와 무관하게 정책을 끊는다.
        #   v1 은 3.0 이었지만 v2 는 기립 후 서 있기/걷기까지 같은 정책이
        #   담당하므로 기본값을 10.0 으로 늘렸다.
        #   0 이하 = 무제한(직접 abort 해야 한다. 권장하지 않는다).
        self.declare_parameter("policy_duration_s", 10.0)

        # ── 학습 cfg 와 반드시 일치시킬 값 ──
        self.declare_parameter("action_scale", 0.25)     # GetupJointPositionActionCfg
        self.declare_parameter("clip_actions", 10.0)
        self.declare_parameter("base_ang_vel_scale", 0.2)
        self.declare_parameter("joint_vel_scale", 0.05)

        # ── [v2 신규] gait 위상 ──
        #   gait_mode:
        #     "stop"  (기본) 완전 정지. freq 0, 위상 0 수렴 → 관측 (1,0,0)
        #                    학습 Play 조건에서 60 % 를 차지한다.
        #     "march"        제자리 걷기. 속도 0 이어도 주파수 유지 (40 %)
        #     "walk"         항상 주파수 유지 (cmd_vel 을 쓸 때)
        #     "auto"         속도 0 이 gait_stop_delay_s 이상 지속되면 정지
        self.declare_parameter("gait_mode", "stop")
        # 학습 frequency_range=(1.7, 2.4) 의 중앙 근처. 범위 밖은 분포 외.
        self.declare_parameter("gait_frequency", 2.0)
        # 학습 frequency_scale=2.0 (관측 셋째 성분 정규화 계수)
        self.declare_parameter("gait_frequency_scale", 2.0)
        # 학습 stop_rate=2.0 [phase/s] → 최대 정지 시간 0.5/2.0 = 0.25 s
        self.declare_parameter("gait_stop_rate", 2.0)
        # 학습 standing_threshold=0.1 (속도 명령 노름 기준 정지 판정)
        self.declare_parameter("gait_standing_threshold", 0.1)
        # auto 모드에서 정지로 넘어가기까지의 유예 [s]
        self.declare_parameter("gait_stop_delay_s", 0.5)
        # PREP 구간에서 위상을 얼릴지. 학습 freeze_during_reset_hold=True.
        self.declare_parameter("gait_freeze_during_prep", True)

        # ── [v2 신규] velocity_commands ──
        #   "zero"      (기본) 항상 (0,0,0). 학습 RobotPlayEnvCfg 와 동일 조건.
        #   "subscribe" /cmd_vel 을 구독한다. 기립 후 보행까지 이어서 볼 때.
        self.declare_parameter("cmd_vel_source", "zero")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        # 이 시간 동안 /cmd_vel 이 안 오면 0 으로 되돌린다(통신 끊김 보호).
        self.declare_parameter("cmd_vel_timeout_s", 0.5)
        # 학습 limit_ranges 로 자른다. [x_lo, x_hi, y_lo, y_hi, wz_lo, wz_hi]
        self.declare_parameter("cmd_vel_limits", CMD_VEL_LIMITS_DEFAULT)
        # 이 각도보다 많이 기울어져 있으면 속도 명령을 0 으로 강제한다.
        #   기상 중(누워 있는 동안)에는 학습 Play 조건이 속도 0 이다.
        #   음수면 비활성.
        self.declare_parameter("cmd_vel_require_upright_deg", 25.0)

        # ── 액션 원점 / 보행 복귀 자세 오버라이드 ──
        #   길이 18 (GETUP_JOINTS 순서). 비우면 위 상수를 쓴다.
        self.declare_parameter("default_joint_pos", [])
        self.declare_parameter("walk_ready_joint_pos", [])

        # ── 안전 게이트 ──
        # 정말 누워 있을 때만 시작하도록 막는다. 서 있는 상태에서 기상
        # 정책을 돌리면 관측이 분포 밖이라 그대로 넘어진다.
        #   ※ v2 는 서 있는 상태도 학습 분포 안이지만(보행 통합), 이 노드는
        #     기상 시험용이므로 기본은 막아 둔다. 서 있는 상태에서 보행을
        #     확인하려면 require_supine:=false 로 풀 것.
        self.declare_parameter("require_supine", True)
        self.declare_parameter("supine_tilt_min_deg", 60.0)
        # projected_gravity 의 x 성분 상한.
        #   실측: 앞으로 엎어짐 gx=+0.90 / 뒤로 누움 gx=-0.96
        #   학습은 supine(등을 댄 자세)만 다루므로 gx 가 음수여야 한다.
        self.declare_parameter("supine_gx_max", -0.5)

        # 목표 변화율 제한. 기본값은 사실상 비활성(기상은 크게 휘저어야 한다).
        # 팔이 튀는 것이 무서우면 0.5 rad 정도로 조여서 첫 시험을 해 볼 것.
        self.declare_parameter("max_target_delta_from_current_rad", 10.0)
        # 관절 한계 클램프 사용 여부. 학습이 항상 클램프하므로 기본 True.
        self.declare_parameter("enable_target_clamp", True)

        # dry_run: 모든 계산·로그는 하되 /joint_goals 를 **발행하지 않는다**.
        #   첫 시험에서 관측/액션 크기만 CSV 로 확인하고 싶을 때 쓴다.
        self.declare_parameter("dry_run", False)

        # ── 트리거 ──
        # 0 이상이면 준비 완료 후 그 시간이 지나면 자동 시작. 음수면 수동.
        # ★ 기본은 수동(-1) ★ 자동 시작은 로봇을 눕히기 전에 팔이 움직인다.
        self.declare_parameter("auto_start_delay_s", -1.0)

        # ── 마무리 동작 ──
        #   hold       : 정책 마지막 목표를 계속 유지 (기본, 가장 안전)
        #   walk_ready : end_transition_s 동안 보행 시작 자세로 코사인 전이
        #   none       : 발행 중단 (모터는 마지막 목표를 그대로 잡고 있는다)
        self.declare_parameter("end_action", "hold")
        self.declare_parameter("end_transition_s", 1.0)

        # ── 하드웨어 준비/복원 ──
        self.declare_parameter("enable_torque_on_start", True)
        # 팔 Goal PWM 을 100% 로 승격. 이게 없으면 팔이 안 움직인다(상한 50).
        self.declare_parameter("raise_arm_goal_pwm", True)
        # 끝나고 motor_gains.yaml 운용값으로 되돌린다(팔 상한 50 복구).
        self.declare_parameter("restore_goal_pwm_on_finish", True)
        # 정책 구간 Profile Velocity [rpm]. -1 이면 건드리지 않는다.
        #   ※ 0 은 "무제한" 특수값이라 절대 넣지 말 것.
        self.declare_parameter("policy_profile_velocity_rpm", -1.0)
        # 위 값을 바꿨을 때 되돌릴 값 [rpm]. robot_init 의 final_rpm 과 동일.
        self.declare_parameter("restore_profile_velocity_rpm", 45.0)

        # ── IMU 보정 (보행 러너와 같은 규약) ──
        #   +x = "앞으로 숙였다고 인지". 기상은 자세 변화가 커서 보행만큼
        #   민감하지 않지만, 필요하면 같은 방식으로 트림할 수 있게 둔다.
        self.declare_parameter("grav_offset_x", 0.0)
        self.declare_parameter("grav_offset_y", 0.0)
        self.declare_parameter("grav_offset_z", 0.0)
        self.declare_parameter("gyro_bias", [0.0, 0.0, 0.0])

        # ── 디버그 / 로그 ──
        self.declare_parameter("debug_print_every", 25)
        self.declare_parameter("debug_frame", "base_link")
        self.declare_parameter("publish_markers", True)
        self.declare_parameter("enable_csv_log", True)
        self.declare_parameter("csv_output_dir", "~/policy_logs")
        self.declare_parameter("csv_flush_every", 25)

        # ════════════════════════════════════════════════════════════
        #  파라미터 읽기
        # ════════════════════════════════════════════════════════════
        _actor_param = str(self.get_parameter("actor_path").value).strip()
        if not _actor_param:
            # 빈 문자열이면 expanduser("") 가 홈 디렉터리로 해석되어
            # "actor_path not found: /home/user" 같은 혼란스러운 에러가 난다.
            _actor_param = default_actor
            self.get_logger().warn(
                f"actor_path 가 비어 있어 기본 경로를 사용합니다: {default_actor}")
        self.actor_path = _expand_path(_actor_param)

        self.policy_rate_hz = float(self.get_parameter("policy_rate_hz").value)
        self.policy_dt = 1.0 / max(1e-6, self.policy_rate_hz)

        self.prep_duration_s = float(self.get_parameter("prep_duration_s").value)
        self.prep_settle_s = float(self.get_parameter("prep_settle_s").value)
        self.policy_duration_s = float(self.get_parameter("policy_duration_s").value)

        self.action_scale = float(self.get_parameter("action_scale").value)
        self.clip_actions = float(self.get_parameter("clip_actions").value)
        self.base_ang_vel_scale = float(self.get_parameter("base_ang_vel_scale").value)
        self.joint_vel_scale = float(self.get_parameter("joint_vel_scale").value)

        # ── gait ──
        self.gait_mode = str(self.get_parameter("gait_mode").value).strip().lower()
        if self.gait_mode not in GAIT_MODES:
            self.get_logger().warn(
                f"gait_mode='{self.gait_mode}' 는 알 수 없는 값 — 'stop' 으로 강제합니다. "
                f"(허용: {list(GAIT_MODES)})")
            self.gait_mode = "stop"
        self.gait_frequency = float(self.get_parameter("gait_frequency").value)
        self.gait_frequency_scale = float(
            self.get_parameter("gait_frequency_scale").value)
        self.gait_stop_rate = float(self.get_parameter("gait_stop_rate").value)
        self.gait_standing_threshold = float(
            self.get_parameter("gait_standing_threshold").value)
        self.gait_stop_delay_s = float(self.get_parameter("gait_stop_delay_s").value)
        self.gait_freeze_during_prep = bool(
            self.get_parameter("gait_freeze_during_prep").value)
        self._warn_gait_frequency(self.gait_frequency)

        # ── cmd_vel ──
        self.cmd_vel_source = str(
            self.get_parameter("cmd_vel_source").value).strip().lower()
        if self.cmd_vel_source not in ("zero", "subscribe"):
            self.get_logger().warn(
                f"cmd_vel_source='{self.cmd_vel_source}' 는 알 수 없는 값 — "
                "'zero' 로 강제합니다.")
            self.cmd_vel_source = "zero"
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.cmd_vel_timeout_s = float(self.get_parameter("cmd_vel_timeout_s").value)
        lim = np.array(
            self.get_parameter("cmd_vel_limits").value, dtype=np.float32).reshape(-1)
        if lim.size != 6:
            self.get_logger().warn(
                f"cmd_vel_limits 길이가 {lim.size} 입니다(기대 6). 기본값을 씁니다.")
            lim = np.array(CMD_VEL_LIMITS_DEFAULT, dtype=np.float32)
        self.cmd_lo = np.array([lim[0], lim[2], lim[4]], dtype=np.float32)
        self.cmd_hi = np.array([lim[1], lim[3], lim[5]], dtype=np.float32)
        self.cmd_vel_require_upright_deg = float(
            self.get_parameter("cmd_vel_require_upright_deg").value)

        self.require_supine = bool(self.get_parameter("require_supine").value)
        self.supine_tilt_min_deg = float(
            self.get_parameter("supine_tilt_min_deg").value)
        self.supine_gx_max = float(self.get_parameter("supine_gx_max").value)

        self.max_target_delta = float(
            self.get_parameter("max_target_delta_from_current_rad").value)
        self.enable_target_clamp = bool(
            self.get_parameter("enable_target_clamp").value)
        self.dry_run = bool(self.get_parameter("dry_run").value)
        self.auto_start_delay_s = float(
            self.get_parameter("auto_start_delay_s").value)

        self.end_action = str(self.get_parameter("end_action").value).strip().lower()
        if self.end_action not in ("hold", "walk_ready", "none"):
            self.get_logger().warn(
                f"end_action='{self.end_action}' 는 알 수 없는 값 — 'hold' 로 강제합니다.")
            self.end_action = "hold"
        self.end_transition_s = float(self.get_parameter("end_transition_s").value)
        # 런치에서 준 원래 설정. _tick_finish 가 self.end_action 을 런타임에
        # 바꾸므로(walk_ready → hold), 재시작 때 되돌릴 원본을 따로 둔다.
        self._end_action_cfg = self.end_action

        self.enable_torque_on_start = bool(
            self.get_parameter("enable_torque_on_start").value)
        self.raise_arm_goal_pwm = bool(
            self.get_parameter("raise_arm_goal_pwm").value)
        self.restore_goal_pwm_on_finish = bool(
            self.get_parameter("restore_goal_pwm_on_finish").value)
        self.policy_profile_velocity_rpm = float(
            self.get_parameter("policy_profile_velocity_rpm").value)
        self.restore_profile_velocity_rpm = float(
            self.get_parameter("restore_profile_velocity_rpm").value)

        self.grav_offset = np.array([
            float(self.get_parameter("grav_offset_x").value),
            float(self.get_parameter("grav_offset_y").value),
            float(self.get_parameter("grav_offset_z").value),
        ], dtype=np.float32)
        self.gyro_bias = np.array(
            self.get_parameter("gyro_bias").value, dtype=np.float32).reshape(-1)
        if self.gyro_bias.size != 3:
            self.gyro_bias = np.zeros(3, dtype=np.float32)

        self.debug_print_every = int(self.get_parameter("debug_print_every").value)
        self.debug_frame = str(self.get_parameter("debug_frame").value)
        self.publish_markers = bool(self.get_parameter("publish_markers").value)

        # 런타임 튜닝(ros2 param set)을 받으려면 콜백 등록이 필요하다.
        # 등록하지 않으면 __init__ 에서 캐시한 값이 그대로 고정된다.
        self.add_on_set_parameters_callback(self._on_set_parameters)

        # ── 관측 규격 ──
        self.frame_slices, self.frame_dim = make_frame_spec()      # 66
        self.obs_dim = self.frame_dim * HISTORY_LEN                # 330

        # ── 기본 자세 배열 만들기 ──
        self.default_q = self._resolve_pose_param(
            "default_joint_pos", GETUP_DEFAULT_Q, "학습 기본자세(액션 원점)")
        self.walk_ready_q = self._resolve_pose_param(
            "walk_ready_joint_pos", _load_walk_ready_pose(), "보행 복귀 자세")
        self.default_qd = np.zeros((NUM_JOINTS,), dtype=np.float32)

        # ── 목표 클램프 배열 ──
        self.clamp_lo = np.array(
            [TARGET_LIMITS[j][0] for j in GETUP_JOINTS], dtype=np.float32)
        self.clamp_hi = np.array(
            [TARGET_LIMITS[j][1] for j in GETUP_JOINTS], dtype=np.float32)

        # ── ONNX 로드 ──
        if not os.path.isfile(self.actor_path):
            raise RuntimeError(f"actor_path not found: {self.actor_path}")

        providers = self._select_providers(
            str(self.get_parameter("onnx_provider").value).lower())
        self.session = ort.InferenceSession(self.actor_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        in_shape = self.session.get_inputs()[0].shape
        out_shape = self.session.get_outputs()[0].shape
        self.get_logger().info(f"ONNX loaded: {self.actor_path}")
        self.get_logger().info(f"ONNX provider: {self.session.get_providers()[0]}")
        self.get_logger().info(f"  input:  {self.input_name} {in_shape}")
        self.get_logger().info(f"  output: {self.output_name} {out_shape}")

        # 관측 차원 검증. v1 모델(300)이나 보행 모델(235/240)을 잘못 물리면
        # 여기서 걸린다. 이 검사가 없으면 onnxruntime 이 shape 에러를
        # 매 틱 던지는 형태로 나타나 원인 파악이 느려진다.
        try:
            model_dim = int(in_shape[-1])
            if model_dim != self.obs_dim:
                hint = ""
                if model_dim == 300:
                    hint = ("  → 관측 300 은 기상 V1 정책입니다. "
                            "actor_onnx_getup_runner 로 실행하십시오.\n")
                elif model_dim in (235, 240):
                    hint = ("  → 관측 235/240 은 보행 정책입니다. "
                            "actor_onnx_walking_runner 로 실행하십시오.\n")
                raise RuntimeError(
                    f"ONNX 입력 차원 불일치: 모델={model_dim}, 기대={self.obs_dim}\n"
                    "  기상 V2 정책(Unitree-MX3-Getup-V2)은 관측 330 "
                    "(프레임 66 x 히스토리 5) 입니다.\n" + hint
                )
        except (TypeError, ValueError):
            self.get_logger().warn("ONNX 입력 차원을 확인할 수 없습니다(동적 shape).")

        # ── 상태 변수 ──
        self.state = ST_WAIT_READY
        self.state_since = time.time()          # 현재 상태 진입 시각
        self.ready_since: Optional[float] = None  # 센서가 처음 갖춰진 시각

        self.last_joint_state: Optional[JointState] = None
        self.last_imu: Optional[Imu] = None
        self.last_gravity: Optional[Vector3Stamped] = None

        self.q = np.zeros((NUM_JOINTS,), dtype=np.float32)
        self.qd = np.zeros((NUM_JOINTS,), dtype=np.float32)
        self.last_action = np.zeros((NUM_JOINTS,), dtype=np.float32)

        # /cmd_vel 원본과 실제 관측에 들어간 값(게이트·클램프 적용 후)
        self.cmd_raw = np.zeros((3,), dtype=np.float32)
        self.cmd_used = np.zeros((3,), dtype=np.float32)
        self._cmd_stamp: Optional[float] = None
        self._cmd_gated_reason = ""

        self.gait = PhaseOnlyGaitIntegrator(stop_rate=self.gait_stop_rate)
        self._standing_since: Optional[float] = None

        self.prep_start_q = self.default_q.copy()   # PREP 시작 시점 자세
        self.final_q = self.default_q.copy()        # 정책 마지막 목표
        self.finish_start_q = self.default_q.copy()  # FINISH 전이 시작 자세

        self.history = FrameHistory(H=HISTORY_LEN, dim=self.frame_dim)
        self.history_initialized = False

        self.tick_count = 0
        self.policy_tick = 0
        self._pv_changed = False          # Profile Velocity 를 바꿨는지
        self._arm_pwm_raised = False      # 팔 Goal PWM 을 100% 로 올렸는지
        self._warned_robot_state = False
        self._last_inference_ms = 0.0
        self._tilt_deg = 0.0

        # tick 주기 감시. gait 위상 적분이 policy_dt(고정)를 쓰므로,
        # 실제 tick 이 50 Hz 가 아니면 위상이 학습과 다른 속도로 돈다.
        self._tick_prev_wall: Optional[float] = None
        self._tick_dt_acc = 0.0
        self._tick_dt_n = 0
        self._tick_rate_last_log = 0.0

        # ── CSV ──
        self.enable_csv_log = bool(self.get_parameter("enable_csv_log").value)
        self.csv_flush_every = int(self.get_parameter("csv_flush_every").value)
        self._csv_file = None
        self._csv_writer = None
        self._csv_row_count = 0
        self._csv_closed = False
        if self.enable_csv_log:
            self._init_csv_logger()

        # ── 구독 ──
        #   ★ tick 은 /joint_states 이벤트로 돈다 ★ 타이머로 돌리면 모터
        #     읽기 주기(50Hz)와 위상이 어긋나 관측이 한 프레임씩 낡는다.
        self.create_subscription(JointState, "/joint_states", self.cb_joint, 20)
        self.create_subscription(Imu, "/imu/data", self.cb_imu, 20)
        self.create_subscription(Bool, "/getup_trigger", self.cb_trigger, 10)
        self.create_subscription(Bool, "/getup_abort", self.cb_abort_topic, 10)
        # fall_manager 가 살아 있으면 경고만 띄운다(막지는 않는다).
        self.create_subscription(String, "/robot_state", self.cb_robot_state, 10)

        # gait 모드/주파수 런타임 변경.
        #   ★ 보행 러너의 /gait_mode 와 이름을 겹치지 않게 네임스페이스를
        #     둔다. 두 노드를 동시에 띄우는 일은 없어야 하지만, 실수로
        #     겹쳤을 때 서로의 모드를 바꾸는 사고를 막는다.
        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(
            String, "/getup_v2/gait_mode", self.cb_gait_mode, latched_qos)
        self.create_subscription(
            Float32, "/getup_v2/gait_frequency", self.cb_gait_frequency, latched_qos)

        # 속도 명령 (cmd_vel_source="subscribe" 일 때만 건다)
        if self.cmd_vel_source == "subscribe":
            self.create_subscription(Twist, self.cmd_vel_topic, self.cb_cmd_vel, 10)
            self.get_logger().warn(
                f"[getup2] cmd_vel_source=subscribe — {self.cmd_vel_topic} 을 "
                "관측에 넣습니다. 학습 Play 조건(속도 0)과 다른 시험입니다.")

        # ── 발행 ──
        self.goal_pub = self.create_publisher(JointState, "/joint_goals", 10)
        self.done_pub = self.create_publisher(Bool, "/getup_done", 10)
        self.state_pub = self.create_publisher(String, "/getup_v2/state", 10)
        self.elapsed_pub = self.create_publisher(Float32, "/getup_v2/elapsed", 10)
        self.tilt_pub = self.create_publisher(Float32, "/getup_v2/tilt_deg", 10)
        # gait 디버그 — Foxglove/RViz 에서 위상이 실제로 도는지 눈으로 본다.
        self.phase_pub = self.create_publisher(Float32, "/getup_v2/gait/phase", 10)
        self.freq_pub = self.create_publisher(
            Float32, "/getup_v2/gait/effective_frequency", 10)
        self.gait_state_pub = self.create_publisher(
            String, "/getup_v2/gait/state", 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, "/getup_v2/debug/markers", 10)
        # 하드웨어 준비용 (body_motor_controller 가 구독)
        self.pwm_pub = self.create_publisher(
            JointState, "/u2d2/set_joint_goal_pwm", 10)
        self.pv_pub = self.create_publisher(
            JointState, "/u2d2/set_joint_profile_velocity", 10)

        # ── 서비스 ──
        self.create_service(Trigger, "~/start", self.srv_start)
        self.create_service(Trigger, "~/abort", self.srv_abort)
        self.create_service(Trigger, "~/reset", self.srv_reset)

        # ── 서비스 클라이언트 (하드웨어 준비/복원) ──
        self.cli_torque = self.create_client(TorqueEnable, "/u2d2/torque_enable")
        self.cli_reload_gains = self.create_client(
            Trigger, "/body_motor_controller/reload_gains")

        self.get_logger().info(
            f"[getup2] 관측 {self.obs_dim} (프레임 {self.frame_dim} x {HISTORY_LEN}), "
            f"액션 {NUM_JOINTS}, action_scale={self.action_scale}")
        self.get_logger().info(
            "[getup2] 프레임 배치: ang_vel[0:3] grav[3:6] cmd[6:9] gait[9:12] "
            "q[12:30] qd[30:48] a_prev[48:66]")
        self.get_logger().info(
            f"[getup2] gait: mode={self.gait_mode} freq={self.gait_frequency}Hz "
            f"scale={self.gait_frequency_scale} stop_rate={self.gait_stop_rate} "
            f"(최대 정지 시간 {0.5 / max(1e-6, self.gait_stop_rate):.3f}s)")
        self.get_logger().info(
            f"[getup2] cmd_vel: source={self.cmd_vel_source} "
            f"upright_gate={self.cmd_vel_require_upright_deg:.1f}deg")
        self.get_logger().info(
            f"[getup2] 시퀀스: PREP {self.prep_duration_s:.1f}s "
            f"(+settle {self.prep_settle_s:.1f}s) → POLICY "
            + (f"{self.policy_duration_s:.1f}s" if self.policy_duration_s > 0.0
               else "무제한(★안전장치 없음★)")
            + f" → {self.end_action}")
        if self.policy_duration_s <= 0.0:
            self.get_logger().warn(
                "[getup2] ★ policy_duration_s <= 0 — 시간 안전장치가 없습니다 ★ "
                "정책이 발산해도 자동으로 멈추지 않습니다. "
                "~/abort 를 즉시 호출할 수 있는 상태로만 쓰십시오.")
        if self.dry_run:
            self.get_logger().warn(
                "[getup2] dry_run=True — /joint_goals 를 발행하지 않습니다(로그만).")
        self.get_logger().warn(
            "[getup2] ★ fall_manager_node 를 반드시 끄고 테스트하십시오 ★ "
            "누운 상태를 FALLEN 으로 보고 standup 키프레임과 /joint_goals 를 "
            "두고 싸웁니다.")
        self.get_logger().info(
            "[getup2] 대기 중. 로봇을 등 대고 눕힌 뒤 시작하십시오:\n"
            "  ros2 service call /actor_onnx_getup_v2_node/start "
            "std_srvs/srv/Trigger")

    # ══════════════════════════════════════════════════════════════
    #  초기화 헬퍼
    # ══════════════════════════════════════════════════════════════
    def _resolve_pose_param(self, param_name: str, fallback: dict,
                            label: str) -> np.ndarray:
        """길이 18 파라미터가 있으면 그것을, 없으면 dict 를 배열로 만든다."""
        raw = self.get_parameter(param_name).value
        arr = np.array(raw, dtype=np.float32).reshape(-1) if raw else np.zeros(0)
        if arr.size == NUM_JOINTS:
            self.get_logger().info(f"{label}: 파라미터 {param_name} 사용")
            return arr.astype(np.float32)
        if arr.size not in (0, NUM_JOINTS):
            self.get_logger().warn(
                f"{param_name} 길이가 {arr.size} 입니다(기대 {NUM_JOINTS}). 무시합니다.")
        vals = [float(fallback.get(j, 0.0)) for j in GETUP_JOINTS]
        self.get_logger().info(f"{label}: 내장 기본값 사용")
        return np.array(vals, dtype=np.float32)

    def _select_providers(self, mode: str) -> list:
        available = ort.get_available_providers()
        self.get_logger().info(f"ONNX available providers: {available}")
        if mode == "cpu":
            return ["CPUExecutionProvider"]
        if mode == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if mode == "tensorrt":
            return ["TensorrtExecutionProvider", "CUDAExecutionProvider",
                    "CPUExecutionProvider"]
        providers = []
        if "TensorrtExecutionProvider" in available:
            providers.append("TensorrtExecutionProvider")
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers

    def _warn_gait_frequency(self, f: float):
        """학습 frequency_range 밖이면 경고. 막지는 않는다."""
        lo, hi = GAIT_FREQ_TRAINED_RANGE
        if f > 1e-6 and not (lo <= f <= hi):
            self.get_logger().warn(
                f"[gait] {f:.2f}Hz 는 학습 분포({lo}~{hi}Hz) 밖입니다. "
                "성능이 보장되지 않습니다.")

    def _on_set_parameters(self, params):
        """런타임 파라미터 변경. 시퀀스 중에는 타이밍 변경을 막는다."""
        from rcl_interfaces.msg import SetParametersResult

        for p in params:
            if p.name in ("prep_duration_s", "policy_duration_s",
                          "prep_settle_s") and self.state in (ST_PREP, ST_POLICY):
                return SetParametersResult(
                    successful=False,
                    reason="시퀀스 진행 중에는 타이밍을 바꿀 수 없습니다.")
            if p.name == "prep_duration_s":
                self.prep_duration_s = float(p.value)
            elif p.name == "prep_settle_s":
                self.prep_settle_s = float(p.value)
            elif p.name == "policy_duration_s":
                self.policy_duration_s = float(p.value)
            elif p.name == "action_scale":
                self.action_scale = float(p.value)
            elif p.name == "max_target_delta_from_current_rad":
                self.max_target_delta = float(p.value)
            elif p.name == "enable_target_clamp":
                self.enable_target_clamp = bool(p.value)
            elif p.name == "require_supine":
                self.require_supine = bool(p.value)
            elif p.name == "supine_tilt_min_deg":
                self.supine_tilt_min_deg = float(p.value)
            elif p.name == "supine_gx_max":
                self.supine_gx_max = float(p.value)
            elif p.name == "dry_run":
                self.dry_run = bool(p.value)
            elif p.name == "grav_offset_x":
                self.grav_offset[0] = float(p.value)
            elif p.name == "grav_offset_y":
                self.grav_offset[1] = float(p.value)
            elif p.name == "grav_offset_z":
                self.grav_offset[2] = float(p.value)
            elif p.name == "debug_print_every":
                self.debug_print_every = int(p.value)
            elif p.name == "end_action":
                v = str(p.value).strip().lower()
                if v in ("hold", "walk_ready", "none"):
                    self.end_action = v
                    self._end_action_cfg = v
            # ── gait 런타임 튜닝 ──
            elif p.name == "gait_mode":
                v = str(p.value).strip().lower()
                if v in GAIT_MODES:
                    self.gait_mode = v
                else:
                    return SetParametersResult(
                        successful=False,
                        reason=f"gait_mode 는 {list(GAIT_MODES)} 중 하나여야 합니다.")
            elif p.name == "gait_frequency":
                self.gait_frequency = max(0.0, float(p.value))
                self._warn_gait_frequency(self.gait_frequency)
            elif p.name == "gait_stop_rate":
                self.gait_stop_rate = float(p.value)
                self.gait.stop_rate = self.gait_stop_rate
            elif p.name == "gait_stop_delay_s":
                self.gait_stop_delay_s = float(p.value)
            elif p.name == "cmd_vel_require_upright_deg":
                self.cmd_vel_require_upright_deg = float(p.value)

        return SetParametersResult(successful=True)

    # ══════════════════════════════════════════════════════════════
    #  콜백
    # ══════════════════════════════════════════════════════════════
    def cb_joint(self, msg: JointState):
        """/joint_states 수신 → 18 관절이 모두 있으면 1 tick 진행.

        /joint_states 에는 목 2 관절도 함께 실려 오는데, body_motor_controller
        가 20 관절을 한 메시지로 내보내므로 실제로는 매번 조건을 만족한다.
        (다른 퍼블리셔가 부분 메시지를 내는 구성에 대한 방어)
        """
        self.last_joint_state = msg
        if not GETUP_JOINTS_SET.issubset(msg.name):
            return
        self.control_tick()

    def cb_imu(self, msg: Imu):
        """/imu/data 수신 → projected gravity 파생 계산.

        /imu/projected_gravity 를 따로 구독하지 않는다. 그 토픽은
        orientation 만의 순수 함수라 정보량이 0 인 중복 토픽이고,
        200Hz 구독을 하나 더 두면 executor 깨우기가 그대로 CPU 가 된다.
        """
        self.last_imu = msg
        gx, gy, gz = projected_gravity_from_quat(msg.orientation)
        if self.last_gravity is None:
            # 200Hz 로 객체를 새로 만들면 그것도 비용이라 한 번만 할당하고
            # 이후에는 필드만 갱신한다. SingleThreadedExecutor 라 콜백과
            # 소비가 같은 스레드에서 순차 실행되므로 in-place 가 안전하다.
            self.last_gravity = Vector3Stamped()
            self.last_gravity.header.frame_id = "imu_link"
        self.last_gravity.header.stamp = msg.header.stamp
        self.last_gravity.vector.x = gx
        self.last_gravity.vector.y = gy
        self.last_gravity.vector.z = gz

    def cb_cmd_vel(self, msg: Twist):
        """속도 명령. cmd_vel_source="subscribe" 일 때만 구독이 걸린다."""
        self.cmd_raw[0] = float(msg.linear.x)
        self.cmd_raw[1] = float(msg.linear.y)
        self.cmd_raw[2] = float(msg.angular.z)
        self._cmd_stamp = time.time()

    def cb_gait_mode(self, msg: String):
        mode = msg.data.strip().lower()
        if mode not in GAIT_MODES:
            self.get_logger().warn(
                f"[gait_mode] 알 수 없는 모드: {mode} (무시). 허용: {list(GAIT_MODES)}")
            return
        if mode != self.gait_mode:
            self.get_logger().info(f"[gait_mode] {self.gait_mode} -> {mode}")
        self.gait_mode = mode

    def cb_gait_frequency(self, msg: Float32):
        f = max(0.0, float(msg.data))
        self._warn_gait_frequency(f)
        self.gait_frequency = f

    def cb_trigger(self, msg: Bool):
        if msg.data:
            ok, why = self.try_start("/getup_trigger")
            if not ok:
                self.get_logger().warn(f"[getup2] 시작 거부: {why}")

    def cb_abort_topic(self, msg: Bool):
        if msg.data:
            self.abort("/getup_abort 토픽")

    def cb_robot_state(self, msg: String):
        """fall_manager 가 살아 있으면 한 번만 크게 경고한다."""
        if msg.data != "NORMAL" and not self._warned_robot_state:
            self._warned_robot_state = True
            self.get_logger().error(
                f"[getup2] /robot_state='{msg.data}' — fall_manager_node 가 "
                "동작 중인 것으로 보입니다. 이 노드와 /joint_goals 를 두고 "
                "싸우므로 반드시 종료하고 다시 시도하십시오.")

    def srv_start(self, req, res):
        ok, why = self.try_start("서비스 ~/start")
        res.success = bool(ok)
        res.message = why
        return res

    def srv_abort(self, req, res):
        self.abort("서비스 ~/abort")
        res.success = True
        res.message = "중단했습니다. 현재 자세로 유지합니다."
        return res

    def srv_reset(self, req, res):
        """FINISH/ABORT 에서 IDLE 로 되돌린다.

        FINISH(hold) 는 매 틱 마지막 목표를 계속 발행한다 — 즉 이 노드가
        /joint_goals 를 계속 점유한다. 시험을 끝내고 보행 러너를 띄우려면
        그 점유를 놓아야 하는데, 노드를 죽이지 않고 놓는 방법이 필요하다.

        모터는 마지막 Goal Position 을 그대로 잡고 있으므로 힘이 빠지지는
        않는다. 자세가 무너지지 않으니 안전하게 손을 뗄 수 있다.
        """
        if self.state in (ST_PREP, ST_POLICY):
            res.success = False
            res.message = (f"진행 중에는 리셋할 수 없습니다 (state={self.state}). "
                           "먼저 ~/abort 를 호출하십시오.")
            return res

        self._restore_hardware()
        self.end_action = self._end_action_cfg
        self.set_state(ST_IDLE, "서비스 ~/reset")
        res.success = True
        res.message = ("IDLE 로 복귀했습니다. /joint_goals 발행을 멈췄으므로 "
                       "다른 노드가 이어받을 수 있습니다.")
        return res

    # ══════════════════════════════════════════════════════════════
    #  준비 상태 / 센서 처리
    # ══════════════════════════════════════════════════════════════
    def ready(self) -> bool:
        return (self.last_joint_state is not None
                and self.last_imu is not None
                and self.last_gravity is not None)

    def extract_q_qd(self):
        """/joint_states 에서 18 관절을 학습 순서로 뽑는다."""
        js = self.last_joint_state
        name_to_idx = {name: i for i, name in enumerate(js.name)}
        missing = [j for j in GETUP_JOINTS if j not in name_to_idx]
        if missing:
            raise RuntimeError(f"joint_states missing joints: {missing}")

        q = np.zeros((NUM_JOINTS,), dtype=np.float32)
        qd = np.zeros((NUM_JOINTS,), dtype=np.float32)
        for i, jn in enumerate(GETUP_JOINTS):
            idx = name_to_idx[jn]
            if idx >= len(js.position):
                raise RuntimeError(f"joint_states position missing for {jn}")
            q[i] = float(js.position[idx])
            qd[i] = float(js.velocity[idx]) if idx < len(js.velocity) else 0.0
        return q, qd

    def gravity_vec(self) -> np.ndarray:
        return np.array([
            self.last_gravity.vector.x,
            self.last_gravity.vector.y,
            self.last_gravity.vector.z,
        ], dtype=np.float32)

    def tilt_deg(self) -> float:
        """수직으로부터의 기울기 [deg]. 0=직립, 90=완전히 누움.

        fall_manager 와 같은 식이다: tilt = acos(-gz / |g|)
        """
        g = self.gravity_vec()
        n = float(np.linalg.norm(g))
        if n < 1e-6:
            return 0.0
        c = float(np.clip(-g[2] / n, -1.0, 1.0))
        return math.degrees(math.acos(c))

    def _check_tick_rate(self):
        """실측 tick 주기를 감시한다.

        ★ 왜 필요한가 ★
          gait 위상은 policy_dt(파라미터로 고정된 1/50 s)로 적분한다.
          학습 env.step_dt 가 정확히 0.02 s 이므로 그 값을 그대로 쓰는 것이
          맞지만, 실제 /joint_states 가 40 Hz 로 온다면 실세계에서는 위상이
          20 % 느리게 도는 셈이 된다. 그러면 정책이 보는 위상 궤적은
          학습과 같은데 로봇의 실제 리듬만 어긋난다 — 원인 찾기 어려운
          형태의 sim-to-real 갭이라 여기서 미리 경고한다.
        """
        now = time.time()
        if self._tick_prev_wall is not None:
            dt = now - self._tick_prev_wall
            # 스파이크(다른 노드 기동 등)는 평균을 오염시키므로 버린다.
            if 1e-4 < dt < 0.5:
                self._tick_dt_acc += dt
                self._tick_dt_n += 1
        self._tick_prev_wall = now

        if self._tick_dt_n < 100:
            return
        mean_dt = self._tick_dt_acc / self._tick_dt_n
        self._tick_dt_acc = 0.0
        self._tick_dt_n = 0
        measured_hz = 1.0 / max(1e-6, mean_dt)
        if abs(measured_hz - self.policy_rate_hz) / self.policy_rate_hz > 0.1:
            if now - self._tick_rate_last_log >= 10.0:
                self._tick_rate_last_log = now
                self.get_logger().warn(
                    f"[getup2] tick 실측 {measured_hz:.1f}Hz vs "
                    f"policy_rate_hz={self.policy_rate_hz:.1f}Hz — 10% 이상 "
                    "차이납니다. gait 위상이 실세계에서 학습과 다른 속도로 "
                    "돕니다. /joint_states 주기를 확인하거나 policy_rate_hz 를 "
                    "실측값에 맞추십시오.")

    # ══════════════════════════════════════════════════════════════
    #  [v2 신규] 속도 명령 / gait 주파수 결정
    # ══════════════════════════════════════════════════════════════
    def resolve_velocity_command(self, in_hold: bool) -> np.ndarray:
        """이번 스텝의 velocity_commands 관측값 (vx, vy, wz).

        학습 대응:
          - generated_commands_with_reset_hold : hold 동안 0
          - RobotPlayEnvCfg                     : ranges 전부 0, 전 env standing

        실기에서 0 이 아닌 값을 넣는 경우는 cmd_vel_source="subscribe" 뿐이고,
        그때도 아래 세 관문을 모두 통과해야 한다.
        """
        self._cmd_gated_reason = ""

        # (1) PREP(= 학습 reset_hold) 동안은 무조건 0.
        if in_hold:
            self._cmd_gated_reason = "hold"
            return np.zeros(3, dtype=np.float32)

        # (2) 소스가 zero 면 항상 0. 학습 Play 조건과 동일하다.
        if self.cmd_vel_source != "subscribe":
            return np.zeros(3, dtype=np.float32)

        # (3) 통신 워치독 — 명령이 끊기면 0 으로 되돌린다.
        if self._cmd_stamp is None:
            self._cmd_gated_reason = "no_cmd"
            return np.zeros(3, dtype=np.float32)
        if self.cmd_vel_timeout_s > 0.0:
            if (time.time() - self._cmd_stamp) > self.cmd_vel_timeout_s:
                self._cmd_gated_reason = "cmd_timeout"
                return np.zeros(3, dtype=np.float32)

        # (4) 자세 관문 — 아직 누워 있으면 0.
        #   학습 Play 는 기상 구간 내내 속도 0 이다. 누운 채로 전진 명령을
        #   넣는 조합은 학습에 있긴 하지만(85 % env 가 속도 명령을 받는다)
        #   Play 조건이 아니므로 기본은 막는다.
        if self.cmd_vel_require_upright_deg >= 0.0:
            if self._tilt_deg > self.cmd_vel_require_upright_deg:
                self._cmd_gated_reason = "not_upright"
                return np.zeros(3, dtype=np.float32)

        # (5) 학습 limit_ranges 로 클램프. 분포 밖 명령을 넣지 않는다.
        return np.clip(self.cmd_raw, self.cmd_lo, self.cmd_hi).astype(np.float32)

    def resolve_gait_frequency(self, vel_cmd: np.ndarray, in_hold: bool) -> float:
        """이번 스텝에 적용할 gait 주파수 [Hz].

        학습 PhaseOnlyGaitCommand._update_command 의 판정을 실기로 옮긴 것이다.

        학습에서는
            standing  = |velocity_command| < standing_threshold
            full_stop = standing & _want_stop      (_want_stop 은 리셋마다 확률
                                                    rel_stop_when_standing=0.6
                                                    으로 뽑는 난수)
            freq      = 0 if full_stop else base_frequency
        인데, 실기에서 _want_stop 은 난수가 아니라 **상위 계층의 의도**다.
        gait_mode 파라미터/토픽이 그 역할을 한다.
        """
        # hold 구간: 학습 freeze_during_reset_hold 와 동일하게 위상 정지.
        if in_hold and self.gait_freeze_during_prep:
            return 0.0

        speed = float(np.linalg.norm(vel_cmd[:3]))
        standing = speed < self.gait_standing_threshold

        # 속도 0 지속 시간 추적 (auto 모드 전용)
        now = time.time()
        if standing:
            if self._standing_since is None:
                self._standing_since = now
        else:
            self._standing_since = None

        if self.gait_mode == "stop":
            # 완전 정지. 학습 full_stop env (Play 조건의 60 %) 와 같다.
            return 0.0
        if self.gait_mode == "march":
            # 제자리 걷기. 속도 0 이어도 리듬을 유지한다 (Play 조건의 40 %).
            return self.gait_frequency
        if self.gait_mode == "walk":
            # 속도와 무관하게 항상 주파수 유지.
            return self.gait_frequency

        # auto: 속도 0 이 stop_delay_s 이상 지속되면 완전 정지.
        #   그 전까지는 제자리 걷기로 리듬을 유지해 즉시 출발할 수 있게 한다.
        if standing and self._standing_since is not None:
            if (now - self._standing_since) >= self.gait_stop_delay_s:
                return 0.0
        return self.gait_frequency

    # ══════════════════════════════════════════════════════════════
    #  상태 전이
    # ══════════════════════════════════════════════════════════════
    def set_state(self, new_state: str, reason: str = ""):
        if new_state == self.state:
            return
        self.get_logger().info(
            f"[getup2] {self.state} → {new_state}"
            + (f"  ({reason})" if reason else ""))
        self.state = new_state
        self.state_since = time.time()

    def try_start(self, source: str):
        """시퀀스 시작 시도. (성공여부, 사유) 를 돌려준다.

        ★ 재트리거 허용 범위 ★
          PREP/POLICY 는 동작 중이므로 당연히 거부한다. 하지만 FINISH 는
          end_action="hold" 일 때 **무한히 머무는 상태**다. 여기까지 막으면
          한 번 돌린 뒤 노드를 죽였다 다시 띄워야 두 번째 시험을 할 수 있다.
          → FINISH 는 walk_ready 전이가 실제로 진행 중일 때만 거부한다.
            ABORT 도 재시작을 허용한다 — 중단 후 다시 눕혀 재시도하는 것이
            정상적인 시험 흐름이다.
        """
        if self.state in (ST_PREP, ST_POLICY):
            return False, f"이미 진행 중입니다 (state={self.state})"

        if self.state == ST_FINISH and self.end_action == "walk_ready":
            left = self.end_transition_s - (time.time() - self.state_since)
            if left > 0.0:
                return False, (
                    f"보행 시작 자세로 전이 중입니다 (약 {left:.1f}s 남음). "
                    "잠시 후 다시 시도하십시오.")

        if not self.ready():
            miss = []
            if self.last_joint_state is None:
                miss.append("/joint_states")
            if self.last_imu is None:
                miss.append("/imu/data")
            return False, f"센서 미수신: {', '.join(miss)}"

        # ── 안전 게이트: 정말 등을 대고 누워 있는가 ──
        g = self.gravity_vec()
        tilt = self.tilt_deg()
        if self.require_supine:
            if tilt < self.supine_tilt_min_deg:
                return False, (
                    f"누운 자세가 아닙니다 (tilt={tilt:.1f}deg < "
                    f"{self.supine_tilt_min_deg:.1f}). 등을 대고 눕힌 뒤 "
                    "다시 시도하거나 require_supine:=false 로 강제하십시오.")
            if float(g[0]) > self.supine_gx_max:
                return False, (
                    f"supine 이 아닙니다 (gx={g[0]:+.3f} > "
                    f"{self.supine_gx_max:+.3f}). 이 정책은 '등을 댄 자세' 만 "
                    "학습했습니다. 엎드린 자세(gx>0)에서는 동작하지 않습니다.")

        try:
            self.q, self.qd = self.extract_q_qd()
        except Exception as e:                                  # noqa: BLE001
            return False, f"관절 상태를 읽지 못했습니다: {e}"

        # ── 하드웨어 준비 ──
        self._prepare_hardware()

        # PREP 시작 자세 = 지금 자세. 여기서 기본자세로 코사인 보간한다.
        self.prep_start_q = self.q.copy()
        self.last_action = np.zeros((NUM_JOINTS,), dtype=np.float32)
        self.history_initialized = False
        self.policy_tick = 0
        # ★ gait 위상 초기화 ★
        #   학습 reset_phase_on_resample=True 와 같다. 시퀀스를 새로
        #   시작할 때는 반드시 위상 0(양발 stance)에서 출발해야 한다.
        #   안 하면 이전 시험의 위상이 남아 첫 프레임부터 어긋난다.
        self.gait.reset()
        self._standing_since = None
        # _tick_finish 가 walk_ready 전이를 끝내면 end_action 을 hold 로
        # 덮어쓴다. 재시작 시에는 런치에서 준 설정값으로 되돌려야 두 번째
        # 시험도 같은 마무리 동작을 한다.
        self.end_action = self._end_action_cfg

        self.set_state(ST_PREP, f"{source}, tilt={tilt:.1f}deg gx={g[0]:+.3f}")
        self.get_logger().info(
            "[getup2] PREP 시작 — 현재 자세에서 학습 기본자세로 "
            f"{self.prep_duration_s:.1f}s 동안 이동합니다. "
            f"팔꿈치 {self.prep_start_q[14]:+.2f} → {self.default_q[14]:+.2f} rad")
        return True, "시작했습니다."

    def abort(self, reason: str):
        """즉시 중단. 현재 측정 자세를 목표로 굳혀서 더 이상 안 움직이게 한다."""
        if self.state in (ST_IDLE, ST_WAIT_READY, ST_ABORT):
            self.get_logger().info(f"[getup2] 중단 요청({reason}) — 진행 중이 아닙니다.")
            return
        try:
            q_now, _ = self.extract_q_qd()
        except Exception:                                       # noqa: BLE001
            q_now = self.final_q.copy()
        self.final_q = q_now.copy()
        self.publish_joint_goals(q_now)
        self._restore_hardware()
        self.set_state(ST_ABORT, reason)
        self.get_logger().warn(
            f"[getup2] ★중단★ ({reason}) — 현재 자세로 유지합니다. "
            "토크를 끄려면 정지 버튼을 쓰십시오.")

    # ══════════════════════════════════════════════════════════════
    #  하드웨어 준비 / 복원
    # ══════════════════════════════════════════════════════════════
    def _prepare_hardware(self):
        """토크 ON → **응답을 받은 뒤** 팔 Goal PWM 승격 → Profile Velocity.

        ★ 순서가 핵심이다 (2026-08-29 실기에서 발견) ★
          body_motor_controller 의 /u2d2/torque_enable 핸들러는 토크를 켜기
          "전" 에 _apply_gains(reason="torque_on") 을 부른다. 그 안에서
          apply_goal_pwm_profile() 이 motor_gains.yaml 의 arms.goal_pwm(=50)
          을 그대로 다시 쓴다. 이건 버그가 아니라 의도된 동작이다 — Goal PWM
          은 RAM 이라 모터 REBOOT 로 날아가므로, 토크를 켜는 시점에 복구해
          주는 것이 맞다.

          따라서 토크 ON 요청과 PWM 승격을 **동시에 쏘면 안 된다.** 우리가
          올린 885 를 토크 핸들러가 곧바로 50 으로 되돌린다.

          → 서비스 future 의 done_callback 으로 순서를 보장한다. 응답을
            동기로 기다리면(spin_until_future_complete) executor 가 멈춰
            /joint_states 콜백이 밀리므로 콜백 체이닝을 쓴다.
        """
        if self.dry_run:
            # dry_run 은 "로봇을 움직이지 않는다" 는 뜻이다. 토크 ON 은 그
            # 자체로 관절이 마지막 목표로 튀는 동작이므로 건드리지 않는다.
            self.get_logger().warn(
                "[getup2] dry_run — 토크/Goal PWM/Profile Velocity 를 "
                "건드리지 않습니다.")
            return

        if self.enable_torque_on_start:
            if self.cli_torque.service_is_ready():
                req = TorqueEnable.Request()
                req.enable = True
                fut = self.cli_torque.call_async(req)
                fut.add_done_callback(self._on_torque_enabled)
                self.get_logger().info(
                    "[getup2] 토크 ON 요청 — 응답을 받은 뒤 팔 PWM 을 올립니다 "
                    "(토크 핸들러가 게인·PWM 을 재적용하므로 순서가 중요)")
                return
            self.get_logger().warn(
                "[getup2] /u2d2/torque_enable 서비스가 없습니다 — 토크 ON 생략")

        # 토크를 안 켜는 구성에서는 곧바로 상한을 올린다.
        self._apply_motor_limits("토크 요청 없음")

    def _on_torque_enabled(self, future):
        """토크 ON 서비스 응답 콜백.

        이 시점이면 body_motor_controller 의 _apply_gains(torque_on) 이 이미
        끝나 있다. 지금 올려야 우리 값이 살아남는다.
        """
        try:
            res = future.result()
            ok = bool(getattr(res, "success", False))
            msg = str(getattr(res, "message", ""))
        except Exception as e:                                  # noqa: BLE001
            self.get_logger().warn(f"[getup2] 토크 ON 응답 실패: {e}")
            ok, msg = False, str(e)

        if ok:
            self.get_logger().info(f"[getup2] 토크 ON 완료 ({msg})")
        else:
            self.get_logger().warn(
                f"[getup2] 토크 ON 실패 ({msg}) — 그래도 PWM 상한은 올립니다.")
        self._apply_motor_limits("토크 ON 응답 이후")

    def _apply_motor_limits(self, reason: str):
        """팔 Goal PWM 승격 + (선택) Profile Velocity 기입.

        같은 값을 여러 번 써도 무해하다(SyncWrite 6~18 레지스터). 그래서
        토크 응답 직후와 POLICY 진입 직전에 두 번 부른다 — PREP 도중
        누군가 reload_gains 를 부르거나 모터가 REBOOT 되면 상한이 조용히
        50 으로 되돌아가기 때문이다.
        """
        if self.dry_run:
            return

        if self.raise_arm_goal_pwm:
            # effort <= 0 이면 "그 모터의 PWM Limit(=100%)" 를 뜻한다.
            # 팔은 motor_gains.yaml 에서 goal_pwm: 50 (약 5.6%) 으로 묶여
            # 있어, 승격하지 않으면 목표까지 이동하지 못한다(이동 하한 150).
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = list(ARM_JOINTS)
            msg.effort = [0.0] * len(ARM_JOINTS)
            self.pwm_pub.publish(msg)
            self._arm_pwm_raised = True
            self.get_logger().info(
                f"[getup2] 팔 Goal PWM 100% 승격 ({len(ARM_JOINTS)} 관절, {reason}) "
                "— dynamixel_system 터미널에 'Goal PWM (개별):' 로그가 "
                "찍혔는지 확인하십시오.")

        if self.policy_profile_velocity_rpm > 0.0:
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = list(GETUP_JOINTS)
            msg.velocity = [float(self.policy_profile_velocity_rpm)] * NUM_JOINTS
            self.pv_pub.publish(msg)
            self._pv_changed = True
            self.get_logger().info(
                f"[getup2] Profile Velocity {self.policy_profile_velocity_rpm:.1f} rpm "
                f"적용 ({reason})")

    def _restore_hardware(self):
        """운용값 복원. 팔 상한을 안 내리면 다음 보행/낙상 때 보호가 사라진다."""
        if self.dry_run:
            # 애초에 아무것도 안 건드렸으므로 복원할 것도 없다.
            return

        if self.restore_goal_pwm_on_finish and self._arm_pwm_raised:
            if self.cli_reload_gains.service_is_ready():
                self.cli_reload_gains.call_async(Trigger.Request())
                self._arm_pwm_raised = False
                self.get_logger().info(
                    "[getup2] reload_gains 호출 — motor_gains.yaml 운용값 복원 "
                    "(팔 Goal PWM 상한 복구)")
            else:
                self.get_logger().warn(
                    "[getup2] /body_motor_controller/reload_gains 서비스가 "
                    "없습니다.\n"
                    "  ★ 팔 Goal PWM 이 100% 로 남아 있습니다 ★ 수동 복원:\n"
                    "  ros2 service call /body_motor_controller/reload_gains "
                    "std_srvs/srv/Trigger")

        if self._pv_changed and self.restore_profile_velocity_rpm > 0.0:
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = list(GETUP_JOINTS)
            msg.velocity = [
                float(self.restore_profile_velocity_rpm)] * NUM_JOINTS
            self.pv_pub.publish(msg)
            self._pv_changed = False
            self.get_logger().info(
                f"[getup2] Profile Velocity "
                f"{self.restore_profile_velocity_rpm:.1f} rpm 복원")

    # ══════════════════════════════════════════════════════════════
    #  관측 프레임
    # ══════════════════════════════════════════════════════════════
    def one_frame(self, vel_cmd: np.ndarray) -> np.ndarray:
        """관측 1 프레임(66 차원) 생성.

        순서는 학습 getup_v2_env_cfg.ObservationsCfg.PolicyCfg 선언 순서와
        반드시 같다:
            base_ang_vel*0.2 → projected_gravity → velocity_commands
            → gait_phase → joint_pos_rel → joint_vel_rel*0.05 → last_action

        ※ 학습에는 관측 노이즈(Unoise)가 붙어 있지만 배포에서는 넣지 않는다.
          노이즈는 강건성 학습용이지 추론 시 재현 대상이 아니다.
          (gait_phase 는 학습에서도 노이즈가 없다 — 내부 시계라 오차가 없다)
        """
        imu_w = np.array([
            self.last_imu.angular_velocity.x,
            self.last_imu.angular_velocity.y,
            self.last_imu.angular_velocity.z,
        ], dtype=np.float32) - self.gyro_bias
        w_b = (self.base_ang_vel_scale * imu_w).astype(np.float32)

        g_b = (self.gravity_vec() - self.grav_offset).astype(np.float32)

        gait_obs = self.gait.observation(self.gait_frequency_scale)

        q_rel = (self.q - self.default_q).astype(np.float32)
        qd_rel = (self.qd - self.default_qd).astype(np.float32)
        qd_scaled = (self.joint_vel_scale * qd_rel).astype(np.float32)

        frame = np.concatenate(
            [w_b, g_b, vel_cmd, gait_obs, q_rel, qd_scaled, self.last_action],
            axis=0,
        ).astype(np.float32)
        assert frame.shape == (self.frame_dim,), frame.shape
        return frame

    # ══════════════════════════════════════════════════════════════
    #  발행
    # ══════════════════════════════════════════════════════════════
    def publish_joint_goals(self, q_target: np.ndarray):
        """18 관절 목표를 /joint_goals 로 발행한다.

        body_motor_controller 는 관절 단위로 병합하므로, 목(neck) 명령을
        지우지 않는다. 목은 neck_control_node 가 계속 소유한다.
        """
        if self.dry_run:
            return
        msg = JointState()
        if self.last_joint_state is not None:
            msg.header.stamp = self.last_joint_state.header.stamp
        else:
            msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(GETUP_JOINTS)
        msg.position = [float(x) for x in q_target]
        self.goal_pub.publish(msg)

    def publish_gait_debug(self):
        """위상/주파수/상태를 디버그 토픽으로 발행.

        Foxglove 나 RViz 에서 phase 가 톱니파를 그리는지, 정지 명령 후
        0 으로 수렴하는지 눈으로 확인할 수 있다. gait 관측이 v2 신규
        항목이라 **여기가 맞는지부터 확인**해야 하므로 반드시 본다.
        """
        m = Float32()
        m.data = float(self.gait.phase)
        self.phase_pub.publish(m)

        m2 = Float32()
        m2.data = float(self.gait.effective_frequency)
        self.freq_pub.publish(m2)

        s = String()
        speed = float(np.linalg.norm(self.cmd_used[:3]))
        s.data = self.gait.state_str(standing=speed < self.gait_standing_threshold)
        self.gait_state_pub.publish(s)

    def publish_debug(self, elapsed: float):
        """상태/경과/기울기 + RViz 마커.

        RViz/Foxglove 에서 시퀀스가 어느 구간인지, 기울기가 실제로
        줄어드는지(=일어나고 있는지), gait 위상이 도는지를 눈으로 본다.
        """
        m = String()
        m.data = self.state
        self.state_pub.publish(m)

        e = Float32()
        e.data = float(elapsed)
        self.elapsed_pub.publish(e)

        t = Float32()
        t.data = float(self._tilt_deg)
        self.tilt_pub.publish(t)

        self.publish_gait_debug()

        if not self.publish_markers:
            return

        # 색: 회색=대기 / 노랑=PREP / 초록=POLICY / 하늘=FINISH / 빨강=ABORT
        if self.state == ST_POLICY:
            rgba = (0.15, 0.85, 0.30, 0.95)
        elif self.state == ST_PREP:
            rgba = (0.95, 0.80, 0.15, 0.95)
        elif self.state == ST_FINISH:
            rgba = (0.25, 0.70, 0.95, 0.90)
        elif self.state == ST_ABORT:
            rgba = (0.95, 0.15, 0.15, 0.95)
        else:
            rgba = (0.55, 0.55, 0.65, 0.60)

        stamp = self.get_clock().now().to_msg()
        arr = MarkerArray()

        # ── (1) 중력 벡터 화살표 ──
        #   몸이 어느 쪽으로 누워 있는지 한눈에 보인다.
        g = self.gravity_vec()
        arrow = Marker()
        arrow.header.frame_id = self.debug_frame
        arrow.header.stamp = stamp
        arrow.ns = "getup_v2"
        arrow.id = 0
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        scale = 0.25
        arrow.points = [
            Point(x=0.0, y=0.0, z=0.0),
            Point(x=float(g[0]) * scale, y=float(g[1]) * scale,
                  z=float(g[2]) * scale),
        ]
        # ARROW + points 모드: x=자루 지름, y=머리 지름, z=머리 길이
        arrow.scale.x = 0.015
        arrow.scale.y = 0.035
        arrow.scale.z = 0.05
        arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = rgba
        arr.markers.append(arrow)

        # ── (2) 상태 텍스트 ──
        text = Marker()
        text.header.frame_id = self.debug_frame
        text.header.stamp = stamp
        text.ns = "getup_v2"
        text.id = 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.z = 0.40
        text.pose.orientation.w = 1.0
        text.scale.z = 0.055
        text.color.r, text.color.g, text.color.b, text.color.a = rgba
        limit = self._state_limit()
        gate = f" [{self._cmd_gated_reason}]" if self._cmd_gated_reason else ""
        text.text = (
            f"{self.state}  {elapsed:.2f}s"
            + (f" / {limit:.1f}s" if limit > 0.0 else "") + "\n"
            f"tilt {self._tilt_deg:5.1f}deg  "
            f"g=({g[0]:+.2f},{g[1]:+.2f},{g[2]:+.2f})\n"
            f"gait {self.gait_mode} phase={self.gait.phase:.3f} "
            f"f={self.gait.effective_frequency:.2f}Hz\n"
            f"cmd=({self.cmd_used[0]:+.2f},{self.cmd_used[1]:+.2f},"
            f"{self.cmd_used[2]:+.2f}){gate}\n"
            f"infer {self._last_inference_ms:.2f}ms"
            + ("  [DRY RUN]" if self.dry_run else ""))
        arr.markers.append(text)

        # ── (3) gait 위상 다이얼 ──
        #   v2 신규 관측이라 "정말 도는지 / 정지 시 0 으로 수렴하는지" 를
        #   숫자가 아니라 그림으로 확인할 수 있어야 한다. 원(LINE_STRIP)
        #   위를 도는 구(SPHERE)로 그린다. 위상 0 = +x 방향이다.
        dial_r = 0.10
        dial_z = 0.60
        circle = Marker()
        circle.header.frame_id = self.debug_frame
        circle.header.stamp = stamp
        circle.ns = "getup_v2"
        circle.id = 2
        circle.type = Marker.LINE_STRIP
        circle.action = Marker.ADD
        circle.pose.orientation.w = 1.0
        circle.scale.x = 0.004          # LINE_STRIP 은 x 가 선 굵기
        circle.color.r, circle.color.g = 0.6, 0.6
        circle.color.b, circle.color.a = 0.6, 0.5
        circle.points = [
            Point(x=dial_r * math.cos(2.0 * math.pi * k / 36.0),
                  y=dial_r * math.sin(2.0 * math.pi * k / 36.0),
                  z=dial_z)
            for k in range(37)
        ]
        arr.markers.append(circle)

        knob = Marker()
        knob.header.frame_id = self.debug_frame
        knob.header.stamp = stamp
        knob.ns = "getup_v2"
        knob.id = 3
        knob.type = Marker.SPHERE
        knob.action = Marker.ADD
        knob.pose.orientation.w = 1.0
        two_pi_phase = 2.0 * math.pi * self.gait.phase
        knob.pose.position.x = dial_r * math.cos(two_pi_phase)
        knob.pose.position.y = dial_r * math.sin(two_pi_phase)
        knob.pose.position.z = dial_z
        knob.scale.x = knob.scale.y = knob.scale.z = 0.028
        # 보행 중이면 주황, 수렴 중이면 노랑, 완전 정지면 회색.
        if self.gait.is_walking:
            knob.color.r, knob.color.g, knob.color.b = 1.0, 0.55, 0.10
        elif self.gait.is_stopping:
            knob.color.r, knob.color.g, knob.color.b = 0.95, 0.90, 0.20
        else:
            knob.color.r, knob.color.g, knob.color.b = 0.50, 0.50, 0.55
        knob.color.a = 0.95
        arr.markers.append(knob)

        self.marker_pub.publish(arr)

    def _state_limit(self) -> float:
        """현재 상태의 제한 시간 [s]. 마커 표시용."""
        if self.state == ST_PREP:
            return self.prep_duration_s + self.prep_settle_s
        if self.state == ST_POLICY:
            return max(0.0, self.policy_duration_s)
        if self.state == ST_FINISH and self.end_action == "walk_ready":
            return self.end_transition_s
        return 0.0

    # ══════════════════════════════════════════════════════════════
    #  메인 tick
    # ══════════════════════════════════════════════════════════════
    def control_tick(self):
        if not self.ready():
            return

        # 센서가 처음 갖춰진 순간 IDLE 로 넘어간다.
        if self.state == ST_WAIT_READY:
            self.ready_since = time.time()
            self.set_state(ST_IDLE, "센서 준비 완료")

        try:
            self.q, self.qd = self.extract_q_qd()
        except Exception as e:                                  # noqa: BLE001
            self.get_logger().warn(f"extract_q_qd failed: {e}")
            return

        self._tilt_deg = self.tilt_deg()
        self.tick_count += 1
        self._check_tick_rate()
        elapsed = time.time() - self.state_since

        if self.state == ST_IDLE:
            self._tick_idle(elapsed)
        elif self.state == ST_PREP:
            self._tick_prep(elapsed)
        elif self.state == ST_POLICY:
            self._tick_policy(elapsed)
        elif self.state == ST_FINISH:
            self._tick_finish(elapsed)
        elif self.state == ST_ABORT:
            # 중단 상태에서는 마지막으로 굳힌 자세만 계속 유지한다.
            self.publish_joint_goals(self.final_q)

        self.publish_debug(elapsed)

    def _tick_idle(self, elapsed: float):
        """트리거 대기. /joint_goals 를 발행하지 않는다.

        발행하면 다른 노드(예: 사람이 손으로 자세를 잡아 두는 도구)와
        충돌한다. 이 구간에서 로봇 관절은 모터가 마지막 목표를 잡고 있다.
        """
        if self.auto_start_delay_s < 0.0 or self.ready_since is None:
            return
        if (time.time() - self.ready_since) < self.auto_start_delay_s:
            return
        ok, why = self.try_start("자동 시작")
        if not ok:
            # 매 틱 로그가 쏟아지지 않게 자동 시작은 한 번만 시도한다.
            self.auto_start_delay_s = -1.0
            self.get_logger().warn(
                f"[getup2] 자동 시작 실패 — {why}\n"
                "  조건을 만족시킨 뒤 수동으로 트리거하십시오.")

    def _tick_prep(self, elapsed: float):
        """초기자세 만들기. 현재 자세 → 학습 기본자세 코사인 보간.

        ★ 왜 필요한가 ★
          실기에서 뒤로 넘어지면 팔은 접힌 채(팔꿈치 약 3.04 rad)로 굳는다.
          arms 그룹이 damping 위주라 넘어지는 동안 그 자세를 유지하기
          때문이다. 그런데 기상 정책의 액션 원점은 팔꿈치 ±1.5 다.
          target = default + action*0.25 이므로, 원점이 1.5 rad 어긋난 채
          시작하면 정책이 낼 수 있는 목표 범위 전체가 통째로 밀린다.

        ★ 학습 대응 ★
          이 구간은 학습 reset_hold(0.1 s) 에 대응한다. 그래서 여기서도
          속도 명령 0 / gait 주파수 0 을 유지한다(in_hold=True).
          gait 를 굴려 두고 관측만 안 쓰면, POLICY 진입 시 위상이 이미
          한참 돌아간 상태가 되어 학습(항상 위상 0 에서 시작)과 어긋난다.
        """
        # hold 구간의 명령/위상을 학습과 동일하게 유지한다.
        #   여기서 굳이 매 틱 갱신하는 이유: PREP 이 끝나는 순간의 gait
        #   상태가 곧 POLICY 첫 프레임의 관측이기 때문이다.
        self.cmd_used = self.resolve_velocity_command(in_hold=True)
        freq = self.resolve_gait_frequency(self.cmd_used, in_hold=True)
        self.gait.update(self.policy_dt, freq)

        move_t = self.prep_duration_s
        if elapsed >= move_t + self.prep_settle_s:
            self._enter_policy()
            return

        if move_t <= 1e-6:
            q_goal = self.default_q.copy()
        else:
            s = _cos_ease(elapsed / move_t)
            q_goal = (self.prep_start_q
                      + (self.default_q - self.prep_start_q) * s).astype(np.float32)

        q_goal = self._clamp_target(q_goal)
        self.publish_joint_goals(q_goal)
        self.final_q = q_goal.copy()
        self._log_csv_row(phase=ST_PREP, q_target=q_goal)

        if self.debug_print_every > 0 and (self.tick_count % self.debug_print_every) == 0:
            self.get_logger().info(
                f"[PREP] {elapsed:.2f}/{move_t + self.prep_settle_s:.2f}s "
                f"tilt={self._tilt_deg:.1f}deg "
                f"elbow_r={self.q[14]:+.3f}→{q_goal[14]:+.3f}")

    def _enter_policy(self):
        """PREP → POLICY 전이. 히스토리와 last_action 을 초기화한다.

        ★ 히스토리를 현재 프레임으로 가득 채우는 이유 ★
          학습에서도 에피소드 리셋 직후 히스토리 버퍼가 첫 관측으로
          가득 찬다. PREP 끝에서 로봇은 사실상 정지 상태이므로, 5 프레임을
          같은 값으로 채우는 것이 학습 분포와 가장 가깝다.
        ★ last_action = 0 ★
          학습 에피소드 첫 스텝의 last_action 이 0 이다. PREP 구간의
          보간 목표를 액션으로 환산해 넣으면 분포 밖 값이 들어간다.
        ★ gait 위상은 리셋하지 않는다 ★
          try_start 에서 이미 0 으로 리셋했고, PREP 동안 freq=0 으로
          유지했으므로 여기서는 그대로 0 이다. 다시 reset() 하면
          gait_freeze_during_prep=False 로 실험할 때 PREP 에서 굴린
          위상이 지워져 실험 자체가 무의미해진다.
        """
        self.last_action = np.zeros((NUM_JOINTS,), dtype=np.float32)
        # 첫 프레임의 속도 명령은 아직 hold 기준(0)이다. POLICY 첫 틱에서
        # 곧바로 갱신되므로 여기서는 마지막 hold 값을 그대로 쓴다.
        frame = self.one_frame(self.cmd_used)
        self.history.reset(frame)
        self.history_initialized = True
        self.policy_tick = 0

        # ★ 안전망 ★ PREP 도중에 누군가 reload_gains 를 부르거나 모터가
        #   REBOOT 되면 팔 Goal PWM 상한이 조용히 50 으로 되돌아간다. 그러면
        #   추론은 도는데 팔만 안 움직이는, 원인 찾기 어려운 증상이 된다.
        #   같은 값을 다시 쓰는 것은 무해하므로 추론 직전에 한 번 더 올린다.
        self._apply_motor_limits("POLICY 진입 재확인")

        self.set_state(ST_POLICY, "기본자세 도달")
        self.get_logger().info(
            "[getup2] ★POLICY 시작★ "
            + (f"{self.policy_duration_s:.1f}s 동안 추론합니다. "
               if self.policy_duration_s > 0.0 else "무제한 추론합니다. ")
            + f"tilt={self._tilt_deg:.1f}deg gait={self.gait_mode}")

    def _tick_policy(self, elapsed: float):
        """정책 추론 구간. policy_duration_s 가 지나면 무조건 끝낸다."""
        if self.policy_duration_s > 0.0 and elapsed >= self.policy_duration_s:
            self._enter_finish("시간 종료(안전장치)")
            return

        # ── [v2 신규] 명령 갱신 → gait 위상 전진 ──
        #   ★ 프레임을 만들기 "전" 에 갱신해야 학습과 타이밍이 맞는다 ★
        #     학습에서는 CommandManager 가 ObservationManager 보다 먼저
        #     돌기 때문이다. 순서를 바꾸면 위상이 한 스텝 뒤처진다.
        self.cmd_used = self.resolve_velocity_command(in_hold=False)
        freq = self.resolve_gait_frequency(self.cmd_used, in_hold=False)
        self.gait.update(self.policy_dt, freq)

        # ★ 프레임 재구성 ★
        #   v1 러너는 "액션을 낸 뒤" 다음 프레임을 만들어 push 했다. v2 는
        #   프레임 안에 gait/cmd 가 들어가므로, 위상을 전진시킨 직후의
        #   프레임으로 히스토리 마지막 칸을 **갱신**해야 한다. 그래서
        #   추론 직전에 현재 프레임을 push 하는 구조로 바꿨다.
        #   (v1 방식대로 두면 관측에 들어가는 위상이 한 틱 낡는다)
        self.history.push(self.one_frame(self.cmd_used))

        obs = build_obs_from_frames(
            self.history.get_frames(), self.frame_slices, self.obs_dim)
        obs_input = obs.reshape(1, -1).astype(np.float32)

        t0 = time.perf_counter()
        outputs = self.session.run([self.output_name], {self.input_name: obs_input})
        self._last_inference_ms = (time.perf_counter() - t0) * 1000.0

        a_raw = outputs[0].flatten().astype(np.float32)
        if a_raw.shape != (NUM_JOINTS,):
            self.get_logger().error(
                f"actor output dim invalid: {a_raw.shape} (기대 {NUM_JOINTS})")
            self._enter_finish("액션 차원 오류")
            return

        a_used = np.clip(
            a_raw, -self.clip_actions, self.clip_actions).astype(np.float32)

        # ── 목표 = 기본자세 + 액션*스케일 → 관절 한계 클램프 ──
        #   학습 GetupJointPositionAction 과 같은 순서다.
        q_target = (self.default_q + self.action_scale * a_used).astype(np.float32)
        q_target = self._clamp_target(q_target)

        # ── 실기 전용 안전 클램프: 현재 각도 기준 변화량 제한 ──
        #   기본값은 사실상 비활성(10 rad)이다. 기상은 크게 휘저어야 하는
        #   동작이라 조이면 정책이 의도한 궤적을 못 낸다. 첫 시험에서 팔이
        #   튀는 것이 무서우면 0.5 정도로 줄여서 확인만 하고 되돌릴 것.
        if self.max_target_delta > 0.0:
            q_target = np.clip(
                q_target,
                self.q - self.max_target_delta,
                self.q + self.max_target_delta).astype(np.float32)

        self.publish_joint_goals(q_target)
        self.final_q = q_target.copy()
        self._log_csv_row(phase=ST_POLICY, q_target=q_target,
                          a_raw=a_raw, a_clip=a_used)

        # 다음 틱의 관측에 들어갈 last_action 을 갱신한다.
        #   ★ 여기서 프레임을 push 하지 않는다 ★ 다음 틱 시작에서 gait 를
        #     전진시킨 뒤 push 하므로, 여기서 또 push 하면 히스토리에 같은
        #     시각의 프레임이 두 번 들어가 5 프레임이 4 틱을 덮게 된다.
        self.last_action = a_used.copy()
        self.policy_tick += 1

        if self.debug_print_every > 0 and (self.policy_tick % self.debug_print_every) == 0:
            self.get_logger().info(
                f"[POLICY] {elapsed:.2f}"
                + (f"/{self.policy_duration_s:.1f}s " if self.policy_duration_s > 0.0
                   else "s ")
                + f"tilt={self._tilt_deg:5.1f}deg "
                f"phase={self.gait.phase:.3f} f={self.gait.effective_frequency:.2f} "
                f"|a|max={np.abs(a_used).max():.2f} "
                f"infer={self._last_inference_ms:.2f}ms")

    def _clamp_target(self, q: np.ndarray) -> np.ndarray:
        """관절 목표를 USD soft limit(= TARGET_LIMITS)으로 자른다.

        학습 GetupJointPositionAction 은 매 스텝 이 클램프를 건다
        (target_limits=None → asset.data.soft_joint_pos_limits).
        실기에서도 같은 클램프를 걸어야 정책이 학습 때와 같은 목표를 받는다.
        """
        if not self.enable_target_clamp:
            return q.astype(np.float32)
        return np.clip(q, self.clamp_lo, self.clamp_hi).astype(np.float32)

    def _enter_finish(self, reason: str):
        """POLICY → FINISH. 하드웨어 복원 + 결과 로그 + /getup_done 발행."""
        self.finish_start_q = self.final_q.copy()
        self._restore_hardware()
        self.set_state(ST_FINISH, reason)

        # 결과 판정은 기울기로만 한다(실기에는 몸통 높이 센서가 없다).
        stood = self._tilt_deg < 25.0
        verdict = "기립 성공으로 보임" if stood else "아직 누워 있음"
        self.get_logger().info(
            f"[getup2] 정책 종료 ({reason}) — tilt={self._tilt_deg:.1f}deg "
            f"→ {verdict}. 마무리 동작: {self.end_action}")

        msg = Bool()
        msg.data = True
        self.done_pub.publish(msg)

    def _tick_finish(self, elapsed: float):
        """마무리 구간.

        hold       : 정책 마지막 목표를 계속 유지한다. 가장 안전한 기본값.
        walk_ready : 보행 시작 자세로 코사인 전이. 다른 러너(보행 v0_2/v1s)
                     로 넘길 때 팔꿈치(1.5 → 3.04)를 되돌린다.
        none       : 발행을 멈춘다. 다이나믹셀은 마지막 Goal Position 을
                     그대로 잡고 있으므로 힘이 빠지지는 않는다.
        """
        if self.end_action == "none":
            return

        if self.end_action == "hold":
            self.publish_joint_goals(self.final_q)
            self._log_csv_row(phase=ST_FINISH, q_target=self.final_q)
            return

        # walk_ready
        if self.end_transition_s <= 1e-6:
            q_goal = self.walk_ready_q.copy()
        else:
            s = _cos_ease(elapsed / self.end_transition_s)
            q_goal = (self.finish_start_q
                      + (self.walk_ready_q - self.finish_start_q) * s
                      ).astype(np.float32)
        q_goal = self._clamp_target(q_goal)
        self.publish_joint_goals(q_goal)
        self.final_q = q_goal.copy()
        self._log_csv_row(phase=ST_FINISH, q_target=q_goal)

        if elapsed >= self.end_transition_s and self.end_action == "walk_ready":
            # 전이가 끝나면 hold 로 바꿔 같은 자세를 계속 유지한다.
            self.end_action = "hold"
            self.get_logger().info(
                "[getup2] 보행 시작 자세 전이 완료 — 이 자세로 유지합니다. "
                "다른 러너를 띄우기 전에 ~/reset 을 호출하거나 이 노드를 "
                "종료하십시오 (/joint_goals 충돌 방지).")

    # ══════════════════════════════════════════════════════════════
    #  CSV 로거
    # ══════════════════════════════════════════════════════════════
    def _init_csv_logger(self):
        out_dir = os.path.expanduser(self.get_parameter("csv_output_dir").value)
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(out_dir, f"getup_v2_log_{stamp}.csv")

        self._csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)

        header = ["tick", "time_sec", "phase", "phase_elapsed", "tilt_deg"]
        header += ["imu_wx", "imu_wy", "imu_wz"]
        header += ["imu_ax", "imu_ay", "imu_az"]
        header += ["grav_x", "grav_y", "grav_z"]
        # ── v2 신규 컬럼 ──
        #   관측에 실제로 들어간 값과 원본을 둘 다 남긴다. 게이트 때문에
        #   0 이 된 것인지, 원래 0 이었는지를 구분할 수 있어야 한다.
        header += ["cmd_raw_vx", "cmd_raw_vy", "cmd_raw_wz"]
        header += ["cmd_used_vx", "cmd_used_vy", "cmd_used_wz", "cmd_gate"]
        header += ["gait_mode", "gait_phase", "gait_freq_eff",
                   "gait_obs_cos", "gait_obs_sin", "gait_obs_freq"]
        for prefix in ["q", "qd", "q_rel", "qd_scaled", "a_prev",
                       "a_raw", "a_clip", "q_target", "q_default"]:
            for jn in GETUP_JOINTS:
                header.append(f"{prefix}_{jn}")
        header += ["inference_ms", "publishing"]

        self._csv_writer.writerow(header)
        self._csv_file.flush()
        self.get_logger().info(f"CSV logging: {csv_path} ({len(header)} columns)")

    def _log_csv_row(self, phase: str, q_target: np.ndarray,
                     a_raw: Optional[np.ndarray] = None,
                     a_clip: Optional[np.ndarray] = None):
        """한 틱을 기록한다.

        PREP 구간에는 액션이 없으므로 a_raw/a_clip 컬럼이 빈칸이 된다.
        그래도 행 자체는 남긴다 — 초기자세 이동 중 관절이 어떻게 따라왔는지
        (특히 팔꿈치가 실제로 1.5 까지 갔는지)가 이 구간에 있다.
        """
        if not self.enable_csv_log or self._csv_writer is None:
            return
        try:
            elapsed_all = time.time() - self.state_since
            row = [self.tick_count, f"{time.time():.4f}", phase,
                   f"{elapsed_all:.4f}", f"{self._tilt_deg:.3f}"]

            if self.last_imu is not None:
                row += [
                    f"{self.last_imu.angular_velocity.x:.6f}",
                    f"{self.last_imu.angular_velocity.y:.6f}",
                    f"{self.last_imu.angular_velocity.z:.6f}",
                    f"{self.last_imu.linear_acceleration.x:.6f}",
                    f"{self.last_imu.linear_acceleration.y:.6f}",
                    f"{self.last_imu.linear_acceleration.z:.6f}",
                ]
            else:
                row += [""] * 6

            g = self.gravity_vec()
            row += [f"{g[0]:.6f}", f"{g[1]:.6f}", f"{g[2]:.6f}"]

            # v2 신규: 명령 + gait
            row += [f"{v:.6f}" for v in self.cmd_raw]
            row += [f"{v:.6f}" for v in self.cmd_used]
            row += [self._cmd_gated_reason]
            gait_obs = self.gait.observation(self.gait_frequency_scale)
            row += [self.gait_mode,
                    f"{self.gait.phase:.6f}",
                    f"{self.gait.effective_frequency:.6f}",
                    f"{gait_obs[0]:.6f}", f"{gait_obs[1]:.6f}",
                    f"{gait_obs[2]:.6f}"]

            q_rel = (self.q - self.default_q).astype(np.float32)
            qd_scaled = (self.joint_vel_scale
                         * (self.qd - self.default_qd)).astype(np.float32)
            blank = [""] * NUM_JOINTS
            for arr in [self.q, self.qd, q_rel, qd_scaled, self.last_action]:
                row += [f"{v:.6f}" for v in arr]
            row += ([f"{v:.6f}" for v in a_raw] if a_raw is not None else blank)
            row += ([f"{v:.6f}" for v in a_clip] if a_clip is not None else blank)
            row += [f"{v:.6f}" for v in q_target]
            row += [f"{v:.6f}" for v in self.default_q]
            row += [f"{self._last_inference_ms:.3f}", 0 if self.dry_run else 1]

            self._csv_writer.writerow(row)
            self._csv_row_count += 1
            if self._csv_row_count % self.csv_flush_every == 0:
                self._csv_file.flush()
        except Exception:                                       # noqa: BLE001
            pass

    def _close_csv(self):
        if self._csv_closed or self._csv_file is None:
            return
        try:
            self._csv_file.flush()
            os.fsync(self._csv_file.fileno())
            self._csv_file.close()
            self.get_logger().info(f"CSV closed: {self._csv_row_count} rows")
        except Exception:                                       # noqa: BLE001
            pass
        self._csv_closed = True

    def destroy_node(self):
        # 노드가 죽을 때 팔 Goal PWM 이 100% 로 남아 있으면, 다음 낙상에서
        # 감속기 보호가 사라진다. 반드시 되돌리고 나간다.
        try:
            self._restore_hardware()
        except Exception:                                       # noqa: BLE001
            pass
        self._close_csv()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ActorOnnxGetupV2Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:                                       # noqa: BLE001
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:                                       # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
