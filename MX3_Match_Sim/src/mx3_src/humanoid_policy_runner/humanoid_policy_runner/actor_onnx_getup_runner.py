#!/usr/bin/env python3
# actor_onnx_getup_runner.py
#
# ============================================================================
#  supine/prone get-up (누운 상태에서 일어나기) 정책 실기 테스트 러너
# ============================================================================
#  학습 환경 : Unitree-MX3-Getup-V1-2  (기본)
#              unitree_rl_lab/tasks/locomotion/robots/MX3/getup_v1_2_env_cfg.py
#              ※ Unitree-MX3-Getup-V1 모델도 그대로 돌아간다 —
#                v1_2 는 v1 최고 판(OK_16-27-54_smooth_stage3)의 복사본에서
#                출발했고 **관측·액션 규격이 한 글자도 안 바뀌었다**:
#                  PolicyCfg  ang_vel3 + grav3 + q18 + qd18 + a18 = 프레임 60
#                  history 5 → 관측 300, 액션 18, scale 0.25
#                  target_limits = _TARGET_LIMITS (v1 과 동일한 표)
#                바뀐 것은 전부 학습 쪽(리셋 분포·리워드·마찰 DR)이라
#                러너가 알아야 할 것은 아래 "시작 자세" 하나뿐이다.
#  로봇 자산 : mx_bot.ROBOT_CONFIGS["XM_robot04_getup"]  (팔 구동 ON, 18 DOF)
#
# ============================================================================
#  ★ 2026-09-15 — 시작 자세 제한을 기본 해제했다 (v1_2 대응) ★
# ============================================================================
#  【왜 바꿨나 — 편의가 아니라 학습 조건이 바뀌었다】
#    v1_2 의 reset_base_supine 에 인자 두 개가 붙었다:
#
#      prone_ratio = 0.5
#        env 의 절반이 **엎드린 자세(prone)** 로 스폰된다. v1 은 앙와위
#        전용이었고 설계 노트에도 "prone 은 별도 태스크로" 라고 적혀
#        있었지만, 실기에서 앞으로 넘어지는 경우가 실제로 나와 한 정책이
#        둘 다 하도록 바꿨다. (cfg 주석의 V2 13 판 실측: 복와위 기립률이
#        여섯 각도 -90~+90 전부 100 %)
#
#      fall_angle_min_deg = 0.0
#        각도가 스칼라가 아니라 **밴드**다. 레벨 5 에서 |각도| 가
#        0~90 도 균일로 뽑힌다. 즉 거의 서 있는 상태부터 완전히 누운
#        상태까지 전 구간이 학습 분포 안이다.
#        yaw_range 도 ±π 라 어느 방향을 보고 누웠든 상관없다.
#
#    그래서 v1 이 걸어 두었던 두 관문이 이제는 **학습 범위의 절반 이상을
#    막는 잘못된 게이트**가 됐다:
#      - tilt >= 60 deg 요구  → 0~60 도 구간을 막는다
#      - gx <= -0.5 요구      → prone(gx>0) 을 통째로 막는다
#
#  【파라미터】 start_pose_check (구 require_supine 을 대체한다)
#      "none"   (기본) 자세를 보지 않는다. v1_2 처럼 전 구간을 학습한
#                      정책용. 서 있는 상태에서 눌러도 시작한다.
#      "lying"         누워 있을 때만 (tilt >= lying_tilt_min_deg).
#                      앞뒤는 안 가린다. prone 도 통과한다.
#      "supine"        v1 과 같은 동작. 등을 댄 자세만
#                      (tilt >= lying_tilt_min_deg 이고 gx <= supine_gx_max).
#                      **앙와위 전용 정책(v1 원본)을 돌릴 때만 쓸 것.**
#
#    거부하지 않더라도 시작 시점의 자세는 항상 분류해서 로그·마커·CSV 에
#    남긴다 (SUPINE / PRONE / SIDE / UPRIGHT). prone 과 supine 의 성공률을
#    나눠 보려면 그 값이 있어야 한다.
#
#  【자세 무관으로 열면서 같이 알아야 할 것】
#    (1) PREP 은 자세와 무관하게 여전히 필요하다.
#        실기 팔꿈치는 접힌 채(약 3.04 rad) 굳어 있고 학습 액션 원점은
#        ±1.5 다. 서 있든 누워 있든 이 차이는 똑같이 메워야 한다.
#    (2) PREP 구간에는 균형을 잡아 주는 주체가 없다. 순수 위치 제어로
#        2 초간 자세를 바꾼다. **서 있는 상태**에서 트리거하면 그동안
#        팔이 벌어지며 무게중심이 움직이는데 아무도 반응하지 않는다.
#        서 있는 상태에서 시험할 때는 prep_duration_s 를 3~4 초로 늘려
#        천천히 바꾸거나, 사람이 잡아 주는 편이 안전하다.
#    (3) 게이트가 없으므로 **오트리거가 곧 동작**이 된다. 보행 중에
#        /getup_trigger 가 실수로 들어오면 로봇이 그 자리에서 웅크린다.
#        상위 FSM 에서 트리거를 내보내는 조건을 확인할 것.
#
#  ★ 이 노드는 "정해진 시간 동안만 돌고 스스로 멈추는" 벤치 테스트 노드다 ★
#    보행 러너(actor_onnx_walking_runner)처럼 계속 도는 상시 노드가 아니다.
#    시퀀스는 세 구간으로 고정되어 있다.
#
#       [IDLE]  트리거 대기 (/joint_goals 발행 안 함)
#          │  ← /getup_trigger(Bool) 또는 서비스 ~/start
#          ▼
#       [PREP]   prep_duration_s (기본 2.0 s)
#                현재 자세 → **학습 기본자세(액션 원점)** 로 코사인 보간.
#                실기에서 넘어지면 팔꿈치는 접힌 채(약 3.04 rad)로
#                굳는다. 그런데 학습의 액션 원점은 팔꿈치 ±1.5 다.
#                이 구간이 그 차이를 메운다. (getup_v1_2_env_cfg.py 의
#                "낙하 사실성" 주석이 요구하는 절차와 동일)
#          ▼
#       [POLICY] policy_duration_s (기본 3.0 s)  ← **안전장치**
#                ONNX 추론 → /joint_goals. 시간이 다 되면 무조건 종료한다.
#                sim 실측 기립 완료가 1.0~1.2 s 이므로 3 s 면 충분히 여유롭고,
#                정책이 발산해도 3 s 안에 반드시 손을 뗀다.
#          ▼
#       [FINISH] end_action 에 따라 마무리 (hold / walk_ready / none)
#
# ============================================================================
#  관측 규격 (학습 GetupV12ObservationsCfg.PolicyCfg 선언 순서 그대로)
# ============================================================================
#       [0:3]     base_ang_vel        * 0.2
#       [3:6]     projected_gravity
#       [6:24]    joint_pos_rel       (18)
#       [24:42]   joint_vel_rel       * 0.05  (18)
#       [42:60]   last_action         (18)
#
#     프레임 60, history_length 5  →  관측 300
#     액션 18, action_scale 0.25, use_default_offset=True
#
#  ★ 보행 정책과 결정적으로 다른 점 ★
#     1) 관절이 12 개가 아니라 **18 개** (팔 6 개 포함)
#     2) velocity_commands / gait 관측이 **없다** (commands = None)
#     3) 액션 원점(default_joint_pos)이 보행과 다르다 (팔꿈치 ±1.5)
#     → 보행 러너에 이 모델을 물리면 차원부터 안 맞아 로드 시 죽는다.
#
#  ★ Getup V2 와도 호환되지 않는다 ★
#     V2 는 보행 env 베이스라 관측에 velocity_commands(3) + gait_phase(3) 가
#     더 붙어 330 이다. V2 모델은 actor_onnx_getup_v2_runner 로 실행할 것.
#
#  ★ history concat 은 term-major ★
#     Isaac Lab 은 각 term 의 5 프레임을 먼저 이어붙이고 다음 term 으로
#     넘어간다. 프레임을 통째로 5 번 붙이는 방식이 아니다.
#     (보행 러너 build_obs_from_frames 와 완전히 동일한 규약을 쓴다)
#
# ============================================================================
#  실기에서 반드시 챙겨야 하는 것 — 팔이 처음으로 "구동"된다
# ============================================================================
#  기존 정책 러너는 다리 12 개만 명령했고, 팔은 접힌 자세를 유지만 했다.
#  기상 정책은 팔로 바닥을 밀어야 하므로 팔이 실제로 힘을 써야 한다.
#
#  (1) Goal PWM 상한
#      motor_gains.yaml 의 arms 그룹에 goal_pwm: 50 이 걸려 있다.
#      885 = 100% 이므로 50 은 약 5.6% 다. 실측상 1.0 rad 이동에 최소 150 이
#      필요하므로, **이대로면 팔이 아예 안 움직인다.**
#      → 이 노드가 시작 시 /u2d2/set_joint_goal_pwm 으로 팔 6 개를 100% 로
#        올리고, 끝나면 /body_motor_controller/reload_gains 로 되돌린다.
#        (robot_init / standup behavior 가 하는 것과 같은 절차)
#
#  (2) 팔 PID 게인
#      arms 그룹은 P=800, D=0 이다 (자세 유지만 하던 시절의 값).
#      다리는 P=800, D=500 이다. 팔로 체중을 밀 때 D=0 이면 진동할 수 있다.
#      → 이 노드는 게인을 건드리지 않는다. 팔이 떨리면 motor_gains.yaml 의
#        arms 에 position_d_gain 을 넣고 reload_gains 로 반영할 것.
#        (보행에도 영향이 가는 값이라 자동 변경은 하지 않는다)
#
#  (3) Profile Velocity
#      robot_init 이 끝나면 45 rpm(= 4.71 rad/s)이다. sim 실측 기상 관절속도가
#      4.57 rad/s 라 45 rpm 은 거의 딱 한계선이다. 여유가 필요하면
#      policy_profile_velocity_rpm 을 50 정도로 올려 볼 것 (기본값 -1 = 미변경).
#      ※ XM540-W150 은 11.1 V 무부하 50 rpm 이라 그 이상은 의미가 없다.
#      ※ 0 은 다이나믹셀에서 "무제한" 특수값이다. 절대 넣지 말 것.
#
#  (4) fall_manager_node 를 **반드시 끄고** 테스트할 것
#      로봇이 누워 있으면 fall_manager 가 FALLEN 으로 판정해 standup
#      키프레임을 재생하고 /joint_goals 에 보호자세를 직접 쏜다. 이 노드와
#      /joint_goals 를 두고 싸우게 되어 관절이 진동한다.
#      → 이 노드는 /robot_state 를 구독해 NORMAL 이 아닌 값이 오면 경고만
#        띄운다(동작을 막지는 않는다. 막으면 누운 상태에서 영영 못 돈다).
#
# ============================================================================
#  실행
# ============================================================================
#    # 1) 하드웨어 스택 (fall_manager 없이)
#    ros2 launch humanoid_serial_bridge dynamixel_system.launch.py
#
#    # 2) 이 노드
#    ros2 launch humanoid_policy_runner getup_v1_test.launch.py \
#        actor_path:=/home/robo02/models/getup_v1_2.onnx
#
#    # 3) 로봇을 아무 자세로나 두고 트리거 (기본은 자세를 보지 않는다)
#    ros2 service call /actor_onnx_getup_node/start std_srvs/srv/Trigger
#      또는
#    ros2 topic pub --once /getup_trigger std_msgs/Bool "{data: true}"
#
#    # 앙와위 전용 정책(v1 원본)을 돌릴 때만 게이트를 되살린다
#    ros2 launch humanoid_policy_runner getup_v1_test.launch.py \
#        actor_path:=... start_pose_check:=supine
#
#    # 중단 (언제든)
#    ros2 service call /actor_onnx_getup_node/abort std_srvs/srv/Trigger
#
#    # 다시 시험: 자세를 잡고 ~/start 를 다시 호출하면 된다.
#    #   FINISH(hold)/ABORT 에서 재트리거가 허용된다. 노드를 죽일 필요 없다.
#
#    # /joint_goals 점유 해제 (보행 러너로 넘기기 전)
#    ros2 service call /actor_onnx_getup_node/reset std_srvs/srv/Trigger
#
#  CSV: ~/policy_logs/getup_v1_log_YYYYMMDD_HHMMSS.csv
#       (start_pose 컬럼에 시작 자세 분류가 들어간다 — prone/supine 성공률을
#        나눠 보려면 이 컬럼으로 그룹핑할 것)
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

from geometry_msgs.msg import Point, Vector3Stamped
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

# projected gravity 파생 계산. 보행 러너와 같은 헬퍼를 쓴다.
#   /imu/projected_gravity 를 따로 구독하지 않고 /imu/data 의 orientation 에서
#   직접 만든다(값은 imu_parser 가 보내주던 것과 비트 단위로 동일).
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
except Exception:
    get_package_share_directory = None


# ============================================================================
#  학습 환경에서 그대로 옮겨온 상수들
#  ※ 여기 값이 학습과 어긋나면 정책은 조용히 오동작한다(에러가 안 난다).
#    getup_v1_env_cfg.py / mx_bot.py 를 고치면 반드시 이쪽도 같이 고칠 것.
# ============================================================================

# mx_bot.MX_LEG_JOINTS + MX_ARM_JOINTS 와 완전히 같은 순서.
#   GetupV1ActionsCfg / GetupV1ObservationsCfg 가 preserve_order=True 로
#   이 순서를 고정하고 있으므로, 한 칸이라도 어긋나면 관측·액션이 뒤섞인다.
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

    "r_shoulder_pitch_joint": -0.3,
    "r_shoulder_roll_joint": 0.0,
    "r_elbow_pitch_joint": 1.0,       # ← 보행은 2.967
    "l_shoulder_pitch_joint": 0.3,
    "l_shoulder_roll_joint": 0.0,
    "l_elbow_pitch_joint": -1.0,      # ← 보행은 -2.967
}

# ── 관절 목표 클램프 (getup_v1_env_cfg._TARGET_LIMITS 와 동일) ──
#   학습의 DelayedJointPositionAction 이 매 스텝 target 을 이 범위로 자른다.
#   실기에서도 같은 클램프를 걸어야 정책이 학습 때와 같은 목표를 받는다.
#   USD 실측값이고 soft limit == hard limit 이라 그대로 쓴다.
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

# ── 상태 문자열 ──
ST_WAIT_READY = "WAIT_READY"    # 센서(/joint_states, /imu/data) 대기
ST_IDLE = "IDLE"                # 트리거 대기. /joint_goals 발행 안 함
ST_PREP = "PREP"                # 초기자세 만들기
ST_POLICY = "POLICY"            # 정책 추론
ST_FINISH = "FINISH"            # 마무리 (hold / walk_ready 전이)
ST_ABORT = "ABORT"              # 중단 — 현재 자세로 굳힘


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

    ★ 학습의 _WALK_READY_POSE 가 아니라 **실기 robot_init 자세**를 쓴다 ★
      학습은 팔꿈치 ±2.967(sim 기본각)을 목표로 잡지만, 실기에서 보행
      정책이 도는 동안 팔이 실제로 잡고 있는 자세는 robot_init_behavior 의
      arm_init_q(어깨 pitch ±0.2, 팔꿈치 ±3.04)다. 기상 직후 보행으로
      넘길 거라면 실기 쪽 자세로 맞춰 두는 편이 불연속이 작다.
      (차이는 팔꿈치 0.07 rad 정도라 어느 쪽이든 크지 않다)

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
    except Exception:
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
    """관측 프레임 링버퍼. 보행 러너와 완전히 동일한 구현이다.

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
    """프레임 안의 term 별 슬라이스와 총 차원.

    학습 GetupV1ObservationsCfg.PolicyCfg 의 선언 순서 그대로:
        base_ang_vel(3) → projected_gravity(3) → joint_pos_rel(18)
        → joint_vel_rel(18) → last_action(18)
    """
    slices = []
    ofs = 0
    for n in (3, 3, NUM_JOINTS, NUM_JOINTS, NUM_JOINTS):
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


class ActorOnnxGetupNode(Node):
    """기상 정책 실기 테스트 노드."""

    def __init__(self):
        super().__init__("actor_onnx_getup_node")

        # ── 기본 모델 경로 ──
        #   패키지 안 models/getup_v1.onnx 를 기본으로 쓴다.
        #   실제 소스 경로:
        #     ~/ros2_ws/src/humanoid_policy_runner/models/getup_v1.onnx
        #   --symlink-install 로 빌드하면 share/humanoid_policy_runner/models
        #   가 src 를 가리키는 심볼릭 링크가 되므로, **모델 파일만 교체할 때는
        #   재빌드가 필요 없다**. 다만 models/ 에 새 파일을 추가한 직후에는
        #   setup.py 의 glob('models/*') 가 빌드 시점에 평가되므로 한 번은
        #   colcon build 를 돌려야 share 에 등록된다.
        default_actor = "~/getup_v1.onnx"
        if get_package_share_directory is not None:
            try:
                default_actor = os.path.join(
                    get_package_share_directory("humanoid_policy_runner"),
                    "models",
                    "getup_v1.onnx",
                )
            except Exception:
                pass

        # ────────────────────────────────────────────────────────────
        #  파라미터
        # ────────────────────────────────────────────────────────────
        self.declare_parameter("actor_path", default_actor)
        self.declare_parameter("onnx_provider", "auto")
        # 정책 주기 [Hz]. 실제 tick 은 /joint_states(50Hz) 이벤트로 돈다.
        # 이 값은 dt 계산과 로그 표시에만 쓰인다.
        self.declare_parameter("policy_rate_hz", 50.0)

        # ── 시퀀스 타이밍 ──
        # prep : 현재 자세 → 학습 기본자세. 팔꿈치가 3.04 → 1.5 로 크게
        #        움직이므로 너무 짧으면 반동으로 몸통이 굴러간다.
        self.declare_parameter("prep_duration_s", 2.0)
        # prep 도달 후 추가 정지 대기 [s]. 팔이 움직이며 생긴 몸통 흔들림이
        # 가라앉기를 기다린다. 0 이면 곧바로 정책으로 넘어간다.
        self.declare_parameter("prep_settle_s", 0.0)
        # ★ 안전장치 ★ 이 시간이 지나면 성공/실패와 무관하게 정책을 끊는다.
        #   sim 실측 기립 완료 1.0~1.2 s. 3.0 이면 충분한 여유다.
        self.declare_parameter("policy_duration_s", 3.0)

        # ── 학습 cfg 와 반드시 일치시킬 값 ──
        self.declare_parameter("action_scale", 0.25)     # GetupJointPositionActionCfg
        self.declare_parameter("clip_actions", 10.0)
        self.declare_parameter("base_ang_vel_scale", 0.2)
        self.declare_parameter("joint_vel_scale", 0.05)

        # ── 액션 원점 / 보행 복귀 자세 오버라이드 ──
        #   길이 18 (GETUP_JOINTS 순서). 비우면 위 상수를 쓴다.
        self.declare_parameter("default_joint_pos", [])
        self.declare_parameter("walk_ready_joint_pos", [])

        # ── 시작 자세 게이트 ──
        #   ★ 2026-09-15 기본 해제 ★ v1_2 는 prone_ratio=0.5 /
        #     fall_angle_min_deg=0.0 이라 0~90 도 · 앞뒤 양방향이 전부
        #     학습 분포 안이다. v1 의 게이트를 그대로 두면 학습 범위의
        #     절반 이상을 막는다. 자세한 근거는 파일 상단 주석 참조.
        #
        #   "none"   (기본) 자세를 보지 않고 시작한다
        #   "lying"  누워 있을 때만 (앞뒤는 안 가린다 — prone 통과)
        #   "supine" 등을 댄 자세만 (v1 과 같은 동작. 앙와위 전용 정책용)
        self.declare_parameter("start_pose_check", "none")
        # "lying"/"supine" 에서 쓰는 기울기 하한 [deg]. 90 = 완전히 누움.
        #   (구 supine_tilt_min_deg. 이름만 바꿨고 의미는 같다)
        self.declare_parameter("lying_tilt_min_deg", 60.0)
        # "supine" 에서만 쓰는 projected_gravity x 성분 상한.
        #   실측: 앞으로 엎어짐 gx=+0.90 / 뒤로 누움 gx=-0.96
        self.declare_parameter("supine_gx_max", -0.5)
        # 자세 분류 문턱 [deg]. 이보다 덜 기울었으면 UPRIGHT 로 본다.
        #   거부용이 아니라 로그·CSV 표기용이다.
        self.declare_parameter("upright_tilt_max_deg", 30.0)

        # 목표 변화율 제한. 기본값은 사실상 비활성(기상은 크게 휘저어야 한다).
        # 팔이 튀는 것이 무서우면 0.5 rad 정도로 조여서 첫 시험을 해 볼 것.
        self.declare_parameter("max_target_delta_from_current_rad", 10.0)

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
        #   45 rpm = 4.71 rad/s. sim 실측 기상 관절속도 4.57 rad/s.
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

        # ────────────────────────────────────────────────────────────
        #  파라미터 읽기
        # ────────────────────────────────────────────────────────────
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

        self.start_pose_check = str(
            self.get_parameter("start_pose_check").value).strip().lower()
        if self.start_pose_check not in ("none", "lying", "supine"):
            self.get_logger().warn(
                f"start_pose_check='{self.start_pose_check}' 는 알 수 없는 값 — "
                "'none' 으로 강제합니다. (허용: none / lying / supine)")
            self.start_pose_check = "none"
        self.lying_tilt_min_deg = float(
            self.get_parameter("lying_tilt_min_deg").value)
        self.supine_gx_max = float(self.get_parameter("supine_gx_max").value)
        self.upright_tilt_max_deg = float(
            self.get_parameter("upright_tilt_max_deg").value)

        self.max_target_delta = float(
            self.get_parameter("max_target_delta_from_current_rad").value)
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
        self.frame_slices, self.frame_dim = make_frame_spec()      # 60
        self.obs_dim = self.frame_dim * HISTORY_LEN                # 300

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

        # 관측 차원 검증. 보행 정책(240)을 잘못 물리면 여기서 걸린다.
        try:
            model_dim = int(in_shape[-1])
            if model_dim != self.obs_dim:
                raise RuntimeError(
                    f"ONNX 입력 차원 불일치: 모델={model_dim}, 기대={self.obs_dim}\n"
                    "  기상 정책(Unitree-MX3-Getup-V1)은 관측 300 "
                    "(프레임 60 x 히스토리 5) 입니다.\n"
                    "  240/235 라면 보행 정책이므로 actor_onnx_walking_runner "
                    "로 실행하십시오."
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
        # 시작 시점 자세 분류 (SUPINE/PRONE/SIDE/UPRIGHT).
        #   거부용이 아니라 로그·마커·CSV 표기용이다. IDLE 구간에는
        #   아직 정해지지 않았다는 뜻으로 '-' 를 둔다.
        self.start_pose = "-"

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

        # ── 발행 ──
        self.goal_pub = self.create_publisher(JointState, "/joint_goals", 10)
        self.done_pub = self.create_publisher(Bool, "/getup_done", 10)
        self.state_pub = self.create_publisher(String, "/getup_test/state", 10)
        self.elapsed_pub = self.create_publisher(Float32, "/getup_test/elapsed", 10)
        self.tilt_pub = self.create_publisher(Float32, "/getup_test/tilt_deg", 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, "/getup_test/debug/markers", 10)
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
            f"[getup] 관측 {self.obs_dim} (프레임 {self.frame_dim} x {HISTORY_LEN}), "
            f"액션 {NUM_JOINTS}, action_scale={self.action_scale}")
        self.get_logger().info(
            f"[getup] 시퀀스: PREP {self.prep_duration_s:.1f}s "
            f"(+settle {self.prep_settle_s:.1f}s) → POLICY "
            f"{self.policy_duration_s:.1f}s → {self.end_action}")
        if self.dry_run:
            self.get_logger().warn(
                "[getup] dry_run=True — /joint_goals 를 발행하지 않습니다(로그만).")
        self.get_logger().warn(
            "[getup] ★ fall_manager_node 를 반드시 끄고 테스트하십시오 ★ "
            "누운 상태를 FALLEN 으로 보고 standup 키프레임과 /joint_goals 를 "
            "두고 싸웁니다.")
        if self.start_pose_check == "none":
            self.get_logger().info(
                "[getup] 시작 자세 게이트: none — 현재 자세와 무관하게 "
                "시작합니다 (v1_2 는 supine/prone 0~90도를 모두 학습).")
        else:
            self.get_logger().warn(
                f"[getup] 시작 자세 게이트: {self.start_pose_check} — "
                f"tilt >= {self.lying_tilt_min_deg:.0f}deg"
                + (f" 이고 gx <= {self.supine_gx_max:+.2f}"
                   if self.start_pose_check == "supine" else "")
                + " 일 때만 시작합니다.")
        self.get_logger().info(
            "[getup] 대기 중. 자세를 잡은 뒤 시작하십시오:\n"
            "  ros2 service call /actor_onnx_getup_node/start std_srvs/srv/Trigger")

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
            elif p.name == "start_pose_check":
                v = str(p.value).strip().lower()
                if v in ("none", "lying", "supine"):
                    self.start_pose_check = v
                else:
                    return SetParametersResult(
                        successful=False,
                        reason="start_pose_check 는 none/lying/supine 중 하나여야 합니다.")
            elif p.name == "lying_tilt_min_deg":
                self.lying_tilt_min_deg = float(p.value)
            elif p.name == "supine_gx_max":
                self.supine_gx_max = float(p.value)
            elif p.name == "upright_tilt_max_deg":
                self.upright_tilt_max_deg = float(p.value)
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

    def cb_trigger(self, msg: Bool):
        if msg.data:
            ok, why = self.try_start("/getup_trigger")
            if not ok:
                self.get_logger().warn(f"[getup] 시작 거부: {why}")

    def cb_abort_topic(self, msg: Bool):
        if msg.data:
            self.abort("/getup_abort 토픽")

    def cb_robot_state(self, msg: String):
        """fall_manager 가 살아 있으면 한 번만 크게 경고한다."""
        if msg.data != "NORMAL" and not self._warned_robot_state:
            self._warned_robot_state = True
            self.get_logger().error(
                f"[getup] /robot_state='{msg.data}' — fall_manager_node 가 "
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
        /joint_goals 를 계속 점유한다. 기상 시험을 끝내고 보행 러너를
        띄우려면 그 점유를 놓아야 하는데, 노드를 죽이지 않고 놓는 방법이
        필요하다. 이 서비스가 그 역할이다.

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

    def classify_pose(self) -> str:
        """시작 시점의 자세를 분류한다. **거부용이 아니라 기록용이다.**

        v1_2 는 supine 과 prone 을 반반 학습하므로 둘 다 정상 입력이다.
        그런데 CSV 만 보고는 어느 쪽에서 시작했는지 구분하기 번거롭고,
        prone 과 supine 의 성공률은 따로 봐야 의미가 있다(cfg 주석의
        V2 12 판 vs 13 판 비교가 정확히 그 이야기다). 그래서 한 단어로
        박아 둔다.

        판정은 몸통 프레임 중력벡터로 한다:
            tilt  = acos(-gz/|g|)  수직 대비 기울기. 0=직립, 90=완전히 누움
            gx                     +면 배가 아래(엎드림), -면 등이 아래

        UPRIGHT : tilt <= upright_tilt_max_deg (기본 30). 거의 서 있다.
        PRONE   : 누웠고 gx > +0.5   — 엎드림
        SUPINE  : 누웠고 gx < -0.5   — 등을 댐
        SIDE    : 누웠고 그 사이     — 옆으로 누움 / 애매한 중간
        """
        tilt = self.tilt_deg()
        if tilt <= self.upright_tilt_max_deg:
            return "UPRIGHT"
        gx = float(self.gravity_vec()[0])
        if gx > 0.5:
            return "PRONE"
        if gx < -0.5:
            return "SUPINE"
        return "SIDE"

    def check_start_pose(self):
        """start_pose_check 에 따른 시작 허용 판정. (ok, 사유) 를 돌려준다.

        ★ 기본값 "none" 은 아무것도 막지 않는다 ★
          v1_2 의 리셋 분포(prone_ratio=0.5, fall_angle_min_deg=0.0,
          yaw ±π)가 0~90 도 · 앞뒤 · 전 방향을 모두 덮으므로 거를 근거가
          없다. 막으면 학습한 능력의 절반을 못 쓰게 된다.
        """
        tilt = self.tilt_deg()
        gx = float(self.gravity_vec()[0])

        if self.start_pose_check == "none":
            return True, ""

        if tilt < self.lying_tilt_min_deg:
            return False, (
                f"누운 자세가 아닙니다 (tilt={tilt:.1f}deg < "
                f"{self.lying_tilt_min_deg:.1f}, start_pose_check="
                f"{self.start_pose_check}). 자세를 바꾸거나 "
                "start_pose_check:=none 으로 푸십시오.")

        if self.start_pose_check == "supine" and gx > self.supine_gx_max:
            return False, (
                f"supine 이 아닙니다 (gx={gx:+.3f} > {self.supine_gx_max:+.3f}). "
                "start_pose_check=supine 은 **앙와위 전용 정책(v1 원본)** 을 "
                "위한 설정입니다. v1_2 는 prone 도 학습했으므로 "
                "start_pose_check:=none 을 쓰십시오.")

        return True, ""

    # ══════════════════════════════════════════════════════════════
    #  상태 전이
    # ══════════════════════════════════════════════════════════════
    def set_state(self, new_state: str, reason: str = ""):
        if new_state == self.state:
            return
        self.get_logger().info(
            f"[getup] {self.state} → {new_state}"
            + (f"  ({reason})" if reason else ""))
        self.state = new_state
        self.state_since = time.time()

    def try_start(self, source: str):
        """시퀀스 시작 시도. (성공여부, 사유) 를 돌려준다.

        ★ 재트리거 허용 범위 ★
          PREP/POLICY 는 동작 중이므로 당연히 거부한다. 하지만 FINISH 는
          end_action="hold" 일 때 **무한히 머무는 상태**다. 여기까지 막으면
          한 번 돌린 뒤 노드를 죽였다 다시 띄워야 두 번째 시험을 할 수 있다.
          (2026-08-29 실기에서 확인)
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

        # ── 시작 자세 판정 ──
        #   기본(start_pose_check="none")은 막지 않는다. 그래도 어느
        #   자세에서 출발했는지는 반드시 기록한다.
        g = self.gravity_vec()
        tilt = self.tilt_deg()
        ok, why = self.check_start_pose()
        if not ok:
            return False, why
        self.start_pose = self.classify_pose()

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
        # _tick_finish 가 walk_ready 전이를 끝내면 end_action 을 hold 로
        # 덮어쓴다. 재시작 시에는 런치에서 준 설정값으로 되돌려야 두 번째
        # 시험도 같은 마무리 동작을 한다.
        self.end_action = self._end_action_cfg

        self.set_state(
            ST_PREP,
            f"{source}, pose={self.start_pose} tilt={tilt:.1f}deg gx={g[0]:+.3f}")
        self.get_logger().info(
            f"[getup] 시작 자세: {self.start_pose} "
            f"(tilt={tilt:.1f}deg, g=({g[0]:+.2f},{g[1]:+.2f},{g[2]:+.2f}))")
        self.get_logger().info(
            "[getup] PREP 시작 — 현재 자세에서 학습 기본자세로 "
            f"{self.prep_duration_s:.1f}s 동안 이동합니다. "
            f"팔꿈치 {self.prep_start_q[14]:+.2f} → {self.default_q[14]:+.2f} rad")
        if self.start_pose == "UPRIGHT":
            # PREP 은 순수 위치 제어다. 누워 있으면 바닥이 받쳐 주지만
            # 서 있으면 팔이 벌어지는 동안 균형을 잡아 주는 주체가 없다.
            self.get_logger().warn(
                "[getup] ★ 거의 선 상태에서 시작합니다 ★ PREP 구간에는 정책이 "
                "돌지 않아 균형 제어가 없습니다. 팔이 3.04 → 1.5 로 벌어지는 "
                f"{self.prep_duration_s:.1f}s 동안 넘어질 수 있으니, 잡아 주거나 "
                "prep_duration_s 를 3~4 s 로 늘려 천천히 바꾸십시오.")
        return True, "시작했습니다."

    def abort(self, reason: str):
        """즉시 중단. 현재 측정 자세를 목표로 굳혀서 더 이상 안 움직이게 한다."""
        if self.state in (ST_IDLE, ST_WAIT_READY, ST_ABORT):
            self.get_logger().info(f"[getup] 중단 요청({reason}) — 진행 중이 아닙니다.")
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
            f"[getup] ★중단★ ({reason}) — 현재 자세로 유지합니다. "
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
          올린 885 를 토크 핸들러가 곧바로 50 으로 되돌린다. 실기 로그에서
          "팔 Goal PWM 100% 승격" 이 찍혔는데도 실제 레지스터는 50 이었다.
          (robot_init_behavior 는 같은 프로세스 안에서 set_torque_all(True)
           → _raise_goal_pwm(hw) 순서로 동기 호출이라 이 문제가 없다)

          → 서비스 future 의 done_callback 으로 순서를 보장한다. 응답을
            동기로 기다리면(spin_until_future_complete) executor 가 멈춰
            /joint_states 콜백이 밀리므로 콜백 체이닝을 쓴다.
        """
        if self.dry_run:
            # dry_run 은 "로봇을 움직이지 않는다" 는 뜻이다. 토크 ON 은 그
            # 자체로 관절이 마지막 목표로 튀는 동작이므로 건드리지 않는다.
            self.get_logger().warn(
                "[getup] dry_run — 토크/Goal PWM/Profile Velocity 를 "
                "건드리지 않습니다.")
            return

        if self.enable_torque_on_start:
            if self.cli_torque.service_is_ready():
                req = TorqueEnable.Request()
                req.enable = True
                fut = self.cli_torque.call_async(req)
                fut.add_done_callback(self._on_torque_enabled)
                self.get_logger().info(
                    "[getup] 토크 ON 요청 — 응답을 받은 뒤 팔 PWM 을 올립니다 "
                    "(토크 핸들러가 게인·PWM 을 재적용하므로 순서가 중요)")
                return
            self.get_logger().warn(
                "[getup] /u2d2/torque_enable 서비스가 없습니다 — 토크 ON 생략")

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
            self.get_logger().warn(f"[getup] 토크 ON 응답 실패: {e}")
            ok, msg = False, str(e)

        if ok:
            self.get_logger().info(f"[getup] 토크 ON 완료 ({msg})")
        else:
            self.get_logger().warn(
                f"[getup] 토크 ON 실패 ({msg}) — 그래도 PWM 상한은 올립니다.")
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
                f"[getup] 팔 Goal PWM 100% 승격 ({len(ARM_JOINTS)} 관절, {reason}) "
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
                f"[getup] Profile Velocity {self.policy_profile_velocity_rpm:.1f} rpm "
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
                    "[getup] reload_gains 호출 — motor_gains.yaml 운용값 복원 "
                    "(팔 Goal PWM 상한 복구)")
            else:
                self.get_logger().warn(
                    "[getup] /body_motor_controller/reload_gains 서비스가 없습니다.\n"
                    "  ★ 팔 Goal PWM 이 100% 로 남아 있습니다 ★ 수동 복원:\n"
                    "  ros2 service call /body_motor_controller/reload_gains "
                    "std_srvs/srv/Trigger")

        if self._pv_changed and self.restore_profile_velocity_rpm > 0.0:
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = list(GETUP_JOINTS)
            msg.velocity = [float(self.restore_profile_velocity_rpm)] * NUM_JOINTS
            self.pv_pub.publish(msg)
            self._pv_changed = False
            self.get_logger().info(
                f"[getup] Profile Velocity {self.restore_profile_velocity_rpm:.1f} rpm 복원")

    # ══════════════════════════════════════════════════════════════
    #  관측 프레임
    # ══════════════════════════════════════════════════════════════
    def one_frame(self) -> np.ndarray:
        """관측 1 프레임(60 차원) 생성.

        순서는 학습 GetupV1ObservationsCfg.PolicyCfg 선언 순서와 반드시 같다:
            base_ang_vel*0.2 → projected_gravity → joint_pos_rel
            → joint_vel_rel*0.05 → last_action
        ※ 학습에는 관측 노이즈(Unoise)가 붙어 있지만 배포에서는 넣지 않는다.
          노이즈는 강건성 학습용이지 추론 시 재현 대상이 아니다.
        """
        imu_w = np.array([
            self.last_imu.angular_velocity.x,
            self.last_imu.angular_velocity.y,
            self.last_imu.angular_velocity.z,
        ], dtype=np.float32) - self.gyro_bias
        w_b = (self.base_ang_vel_scale * imu_w).astype(np.float32)

        g_b = (self.gravity_vec() - self.grav_offset).astype(np.float32)

        q_rel = (self.q - self.default_q).astype(np.float32)
        qd_rel = (self.qd - self.default_qd).astype(np.float32)
        qd_scaled = (self.joint_vel_scale * qd_rel).astype(np.float32)

        frame = np.concatenate(
            [w_b, g_b, q_rel, qd_scaled, self.last_action], axis=0
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

    def publish_debug(self, elapsed: float):
        """상태/경과/기울기 + RViz 마커.

        Foxglove 나 RViz 에서 시퀀스가 어느 구간인지, 기울기가 실제로
        줄어드는지(=일어나고 있는지)를 눈으로 볼 수 있어야 한다.
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

        if not self.publish_markers:
            return

        # 색: 파랑=대기 / 노랑=PREP / 초록=POLICY / 하늘=FINISH / 빨강=ABORT
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

        # (1) 중력 벡터 화살표 — 몸이 어느 쪽으로 누워 있는지 한눈에 보인다.
        g = self.gravity_vec()
        arrow = Marker()
        arrow.header.frame_id = self.debug_frame
        arrow.header.stamp = stamp
        arrow.ns = "getup_test"
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

        # (2) 상태 텍스트
        text = Marker()
        text.header.frame_id = self.debug_frame
        text.header.stamp = stamp
        text.ns = "getup_test"
        text.id = 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.z = 0.35
        text.pose.orientation.w = 1.0
        text.scale.z = 0.06
        text.color.r, text.color.g, text.color.b, text.color.a = rgba
        limit = self._state_limit()
        text.text = (f"{self.state}  {elapsed:.2f}s"
                     + (f" / {limit:.1f}s" if limit > 0.0 else "") + "\n"
                     f"start {self.start_pose}  now {self.classify_pose()}\n"
                     f"tilt {self._tilt_deg:5.1f}deg  g=({g[0]:+.2f},{g[1]:+.2f},"
                     f"{g[2]:+.2f})\n"
                     f"infer {self._last_inference_ms:.2f}ms"
                     + ("  [DRY RUN]" if self.dry_run else ""))
        arr.markers.append(text)

        self.marker_pub.publish(arr)

    def _state_limit(self) -> float:
        """현재 상태의 제한 시간 [s]. 마커 표시용."""
        if self.state == ST_PREP:
            return self.prep_duration_s + self.prep_settle_s
        if self.state == ST_POLICY:
            return self.policy_duration_s
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
                f"[getup] 자동 시작 실패 — {why}\n"
                "  조건을 만족시킨 뒤 수동으로 트리거하십시오.")

    def _tick_prep(self, elapsed: float):
        """초기자세 만들기. 현재 자세 → 학습 기본자세 코사인 보간.

        ★ 왜 필요한가 ★
          실기에서 뒤로 넘어지면 팔은 접힌 채(팔꿈치 약 3.04 rad)로 굳는다.
          arms 그룹이 damping 위주라 넘어지는 동안 그 자세를 유지하기
          때문이다. 그런데 기상 정책의 액션 원점은 팔꿈치 ±1.5 다.
          target = default + action*0.25 이므로, 원점이 1.5 rad 어긋난 채
          시작하면 정책이 낼 수 있는 목표 범위 전체가 통째로 밀린다.
          getup_v1_env_cfg.py 의 "낙하 사실성" 주석이 요구하는 절차가 이것이다.
        """
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

        q_goal = np.clip(q_goal, self.clamp_lo, self.clamp_hi).astype(np.float32)
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
        """
        self.last_action = np.zeros((NUM_JOINTS,), dtype=np.float32)
        frame = self.one_frame()
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
            f"[getup] ★POLICY 시작★ {self.policy_duration_s:.1f}s 동안 추론합니다. "
            f"tilt={self._tilt_deg:.1f}deg")

    def _tick_policy(self, elapsed: float):
        """정책 추론 구간. policy_duration_s 가 지나면 무조건 끝낸다."""
        if elapsed >= self.policy_duration_s:
            self._enter_finish("시간 종료(안전장치)")
            return

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

        a_used = np.clip(a_raw, -self.clip_actions, self.clip_actions).astype(np.float32)

        # ── 목표 = 기본자세 + 액션*스케일 → 관절 한계 클램프 ──
        #   학습 DelayedJointPositionAction 과 같은 순서다.
        q_target = (self.default_q + self.action_scale * a_used).astype(np.float32)
        q_target = np.clip(q_target, self.clamp_lo, self.clamp_hi).astype(np.float32)

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

        # 다음 프레임은 갱신된 last_action 을 반영해 다시 만든다.
        self.last_action = a_used.copy()
        self.history.push(self.one_frame())
        self.policy_tick += 1

        if self.debug_print_every > 0 and (self.policy_tick % self.debug_print_every) == 0:
            self.get_logger().info(
                f"[POLICY] {elapsed:.2f}/{self.policy_duration_s:.1f}s "
                f"tilt={self._tilt_deg:5.1f}deg "
                f"|a|max={np.abs(a_used).max():.2f} "
                f"infer={self._last_inference_ms:.2f}ms")

    def _enter_finish(self, reason: str):
        """POLICY → FINISH. 하드웨어 복원 + 결과 로그 + /getup_done 발행."""
        self.finish_start_q = self.final_q.copy()
        self._restore_hardware()
        self.set_state(ST_FINISH, reason)

        # 결과 판정은 기울기로만 한다(실기에는 몸통 높이 센서가 없다).
        stood = self._tilt_deg < 25.0
        verdict = "기립 성공으로 보임" if stood else "아직 누워 있음"
        self.get_logger().info(
            f"[getup] 정책 종료 ({reason}) — tilt={self._tilt_deg:.1f}deg "
            f"→ {verdict}. 마무리 동작: {self.end_action}")

        msg = Bool()
        msg.data = True
        self.done_pub.publish(msg)

    def _tick_finish(self, elapsed: float):
        """마무리 구간.

        hold       : 정책 마지막 목표를 계속 유지한다. 가장 안전한 기본값.
        walk_ready : 보행 시작 자세로 코사인 전이. 기상 성공 후 보행
                     정책으로 넘길 때 팔꿈치(1.5 → 3.04)를 되돌린다.
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
        q_goal = np.clip(q_goal, self.clamp_lo, self.clamp_hi).astype(np.float32)
        self.publish_joint_goals(q_goal)
        self.final_q = q_goal.copy()
        self._log_csv_row(phase=ST_FINISH, q_target=q_goal)

        if elapsed >= self.end_transition_s and self.end_action == "walk_ready":
            # 전이가 끝나면 hold 로 바꿔 같은 자세를 계속 유지한다.
            self.end_action = "hold"
            self.get_logger().info(
                "[getup] 보행 시작 자세 전이 완료 — 이 자세로 유지합니다. "
                "보행 러너를 띄우기 전에 이 노드를 종료하십시오 "
                "(/joint_goals 충돌 방지).")

    # ══════════════════════════════════════════════════════════════
    #  CSV 로거
    # ══════════════════════════════════════════════════════════════
    def _init_csv_logger(self):
        out_dir = os.path.expanduser(self.get_parameter("csv_output_dir").value)
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(out_dir, f"getup_v1_log_{stamp}.csv")

        self._csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)

        header = ["tick", "time_sec", "phase", "phase_elapsed", "tilt_deg"]
        # ★ 2026-09-15 신규 ★ 시작 자세와 현재 자세 분류.
        #   v1_2 는 supine/prone 을 반반 학습하므로 성공률을 나눠 봐야
        #   의미가 있다. start_pose 로 그룹핑하면 바로 갈린다.
        header += ["start_pose", "pose_now"]
        header += ["imu_wx", "imu_wy", "imu_wz"]
        header += ["imu_ax", "imu_ay", "imu_az"]
        header += ["grav_x", "grav_y", "grav_z"]
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
            row += [self.start_pose, self.classify_pose()]

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
    node = ActorOnnxGetupNode()
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
