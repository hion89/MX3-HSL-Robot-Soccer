#!/usr/bin/env python3
# actor_onnx_walking_runner.py
#
# [개명 이력] 2026-08-18  actor_onnx_v1s_runner.py → actor_onnx_walking_runner.py
#   · 실행파일 actor_onnx_v1s_runner → actor_onnx_walking_runner
#   · 노드명   actor_onnx_v1s_node   → actor_onnx_walking_node
#   · 클래스   ActorOnnxV1SNode      → ActorOnnxWalkingNode
#   개명 이유: 이 러너는 v0_1/v0_2/v1/v1s 를 policy_variant 로 모두 지원하는
#   다변형 러너인데, 이름이 특정 변형(v1s)을 가리켜 오해를 샀다. 실제
#   control.launch.py 의 policy_variant 기본값은 v0_2 다. 역할(보행)로 이름을
#   바꿔, 보행+킥 러너(actor_onnx_walkkick_v1_runner)와 대비를 명확히 했다.
#   ※ 본문의 "v1s" 는 정책 변형 이름이므로 그대로 둔다(러너 이름과 무관).
#
# v1 / v1s 정책 공용 실기 ONNX 러너.
# actor_onnx_runner.py (v0 전용) 를 기반으로 하되, 학습 환경이 달라진
# 부분을 반영하고 두 변형을 파라미터로 전환할 수 있게 했다.
#
# ============================================================================
# policy_variant 파라미터
# ============================================================================
#   "v0_1": velocity_v0_1_env_cfg (v0 + 보행 위상만)
#             관측 235 = 프레임 47 x 5
#             gait 2 차원 (cos, sin), 게이팅 없음
#             주파수는 속도 명령이 결정 (속도 0 -> 0), 정지 시 위상 0 수렴
#             정지 구분은 "위상이 얼어붙었는가" 를 히스토리가 담당
#             gait_mode / gait_frequency 토픽은 무시된다
#
#   "v0_2": velocity_v0_2_env_cfg (v0_1 + 제자리 걷기)
#             관측 240 = 프레임 48 x 5
#             gait 3 차원 (cos, sin, freq), 게이팅 없음
#             주파수 2.0 고정. gait_mode 로 보행/제자리걷기/정지 전환
#             ★ 학습이 2.0 고정이므로 gait_frequency 를 바꾸면 분포 밖 ★
#
#   "v1"  : velocity_v1_env_cfg  (htwk 이식 1 차)
#             관측 235 = 프레임 47 x 5
#             gait 2 차원 (cos, sin), 정지 시 (0,0) 게이팅
#             주파수는 속도 명령이 결정 (속도 0 -> 0), 위상 동결
#             gait_mode / gait_frequency 토픽은 무시된다
#
#   "v1s" : velocity_v1s_env_cfg (stop-aware gait)  ← 기본값
#             관측 240 = 프레임 48 x 5
#             gait 3 차원 (cos, sin, freq), 게이팅 없음
#             주파수가 속도와 독립 명령, 정지 시 위상 0 으로 수렴
#             gait_mode 로 보행/제자리걷기/정지 전환
#
#   ONNX 입력 차원이 안 맞으면 로드 시 에러로 잡히고, 어느 변형인지
#   힌트를 출력한다. v0 정책(225)은 이 노드로 실행할 수 없다.
#
#   ★ v0_1 과 v1 은 차원이 같다(235) ★
#   위상 의미가 다르므로 반드시 policy_variant 를 정확히 지정해야 한다.
#   차원 검사로는 걸러지지 않아, 잘못 지정하면 조용히 오동작한다.
#
# ============================================================================
# 관측 프레임 레이아웃 (term 순서 = 학습 ObservationsCfg.PolicyCfg 순서)
# ============================================================================
#       [0:3]     base_ang_vel * 0.2
#       [3:6]     projected_gravity
#       [6:9]     velocity_commands (vx, vy, wz)
#       [9:9+G]   gait          G=2(v1) / 3(v1s)
#       [.. +12]  joint_pos_rel
#       [.. +12]  joint_vel_rel * 0.05
#       [.. +12]  last_action
#
#   Isaac Lab 의 history concat 은 "term-major" 다. 즉 각 term 의 5 프레임을
#   먼저 이어붙이고 그다음 term 으로 넘어간다. build_obs_from_frames 참조.
#   프레임을 통째로 5 번 붙이는 방식이 아니므로 틀리면 정책이 전혀 다른
#   값을 보게 된다.
#
# ============================================================================
# 보행 위상은 이 노드가 직접 적분한다
# ============================================================================
#   학습의 CommandTerm 이 하던 일을 그대로 옮긴 것이다.
#   ★ 반드시 "증분 적분" 이어야 한다 ★
#
#       phase = fmod(phase + dt * freq, 1.0)
#
#   절대 시각 기반(phase = fmod(now * freq, 1.0))으로 구현하면 주파수를
#   바꾸는 순간 위상이 점프한다. (HTWK 배포 코드의 실제 버그)
#
#   정지 처리는 변형별로 다르다. GaitPhaseIntegrator docstring 참조.
#   ★ 위상을 0 으로 즉시 리셋하면 안 된다 ★ 두 변형 모두 학습 분포와
#   어긋난다.
#
# ============================================================================
# v1s 의 세 가지 보행 상태
# ============================================================================
#       속도!=0, freq>0  -> 보행
#       속도==0, freq>0  -> 제자리 걷기 (marching in place)
#       속도==0, freq==0 -> 완전 정지
#
#   학습에서는 rel_stop_when_standing 확률로 뒤 두 상태를 나눴다.
#   실기에서는 상위 계층(FSM)이 /gait_mode 토픽으로 지정한다.
#   기본값 "auto" 는 속도 0 이 stop_delay_s 이상 지속되면 완전 정지로
#   전환한다. 공을 기다리는 짧은 순간에는 리듬을 유지하다가, 오래 서
#   있어야 하면 완전히 멈추는 동작이 된다.
#
# ============================================================================
# CSV: ~/policy_logs/<variant>_log_YYYYMMDD_HHMMSS.csv
# ============================================================================

import os
import csv
import math
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Vector3Stamped, Twist

# [2026-08 CPU 최적화] projected gravity 파생 계산 헬퍼.
#   /imu/projected_gravity(200Hz) 구독을 없애고 /imu/data 의 orientation
#   에서 직접 만든다. 원본(imu_parser_node.MadgwickAHRS.gravity_vector_body)
#   과 수식·연산순서가 동일해 값이 비트 단위로 같다.
#   ※ humanoid_policy_runner 는 package.xml 에서 이미 humanoid_serial_bridge
#     를 의존하고 있다(robot_init_behavior import 선례). 추가 의존 불필요.
from humanoid_serial_bridge.imu_math import projected_gravity_from_quat
from std_msgs.msg import String, Float32, Bool

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


# 학습 시 MX_LEG_JOINTS 와 반드시 같은 순서여야 한다.
# (ActionsCfg / ObservationsCfg 에서 preserve_order=True 로 고정한 순서)
POLICY_JOINTS = [
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
]

POLICY_JOINTS_SET = set(POLICY_JOINTS)

HISTORY_LEN = 5

# ============================================================================
# 정책 변형별 관측 규격
# ============================================================================
# v1  : GaitPhaseCommand        — gait 2 차원 (cos, sin), 게이팅 있음
# v1s : StoppableGaitPhaseCommand — gait 3 차원 (cos, sin, freq), 게이팅 없음
#
# 나머지 term 은 동일하며 gait 차원 차이만큼 프레임 크기가 달라진다.
#   v1  프레임 47 -> 관측 235
#   v1s 프레임 48 -> 관측 240
#
# 슬라이스 순서는 학습의 ObservationsCfg.PolicyCfg 선언 순서와 같아야 한다.


def _make_frame_spec(gait_dim: int):
    """gait 차원에 맞춰 프레임 슬라이스와 총 차원을 만든다."""
    slices = []
    ofs = 0
    for n in (3, 3, 3, gait_dim, 12, 12, 12):
        slices.append((ofs, ofs + n))
        ofs += n
    return slices, ofs


VARIANT_SPEC = {
    # variant: gait 관측 차원
    "v0_1": 2,   # v0 + 위상만. 게이팅 없음, 정지 시 위상 0 수렴
    "v0_2": 3,   # v0_1 + 제자리 걷기. 주파수 관측 포함 (2.0 고정)
    "v1": 2,     # htwk 이식 1차. 게이팅 있음, 정지 시 위상 동결
    "v1s": 3,    # stop-aware. 게이팅 없음, 주파수 가변(1.8~2.2)
}

# 차원이 겹치는 조합이 둘 있다. 위상/주파수 의미가 다르므로 반드시
# policy_variant 를 정확히 지정해야 한다. 차원 검사로는 걸러지지 않는다.
#   235 : v0_1 (위상 0 수렴, 게이팅 없음) vs v1 (위상 동결, 게이팅 있음)
#   240 : v0_2 (주파수 2.0 고정)          vs v1s (주파수 1.8~2.2 가변)


def _expand_path(p: str) -> str:
    return str(Path(os.path.expanduser(p)).resolve())


def _load_default_joint_pos() -> dict:
    """robot_init_behavior 의 leg_init_q 를 기본 자세로 사용."""
    try:
        from humanoid_serial_bridge.behaviors.robot_init_behavior import RobotInitBehavior
        rib = RobotInitBehavior(hw=None)
        return {k: float(v) for k, v in rib.leg_init_q.items()}
    except Exception:
        pass

    return {
        "r_pelvis_pitch_joint": 0.8,
        "r_pelvis_roll_joint": 0.0,
        "r_pelvis_yaw_joint": 0.0,
        "r_knee_pitch_joint": 1.4,
        "r_ankle_pitch_joint": 0.7,
        "r_ankle_roll_joint": 0.0,

        "l_pelvis_pitch_joint": -0.8,
        "l_pelvis_roll_joint": 0.0,
        "l_pelvis_yaw_joint": 0.0,
        "l_knee_pitch_joint": -1.4,
        "l_ankle_pitch_joint": -0.7,
        "l_ankle_roll_joint": 0.0,
    }


class FrameHistory:
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


def build_obs_from_frames(frames: np.ndarray, slices, obs_dim: int) -> np.ndarray:
    """프레임 히스토리를 1 차원 관측 벡터로 변환.

    Isaac Lab 의 ObsGroup(history_length=5, concatenate_terms=True) 는
    "term-major" 로 이어붙인다. 즉 각 term 의 5 프레임을 먼저 나열하고
    다음 term 으로 넘어간다. 프레임을 통째로 5 번 이어붙이는(frame-major)
    방식이 아니므로 순서를 틀리면 정책이 전혀 다른 값을 보게 된다.
    """
    h, dim = frames.shape
    assert h == HISTORY_LEN, frames.shape

    parts = []
    for lo, hi in slices:
        parts.append(frames[:, lo:hi].reshape(h * (hi - lo)))

    obs = np.concatenate(parts, axis=0).astype(np.float32)
    assert obs.shape == (obs_dim,), obs.shape
    return obs


class GaitPhaseIntegrator:
    """보행 위상 적분기. v1 / v1s 두 변형을 모두 지원한다.

    학습 코드의 _update_command() 와 동일한 식을 쓴다. 한 줄이라도
    달라지면 sim-to-real 갭이 생기므로 수정 시 양쪽을 함께 고칠 것.

    ── v1 (htwk_gait.GaitPhaseCommand) ──
      - 주파수는 속도 명령으로 강제 게이팅 (속도 0 -> freq 0)
      - freq 0 이면 위상이 "그 자리에 얼어붙는다" (phase += 0)
      - 관측 2 차원 (cos, sin), 정지 시 (0, 0) 으로 게이팅
        → 단위원 밖 값이라 보행 중 위상과 구분된다

    ── v1s (htwk_gait_stop.StoppableGaitPhaseCommand) ──
      - 주파수가 속도와 독립된 명령 (제자리 걷기 표현 가능)
      - freq 0 이면 위상이 0 을 향해 최단 경로로 수렴
      - 관측 3 차원 (cos, sin, freq), 게이팅 없음
        → 정지 상태는 freq=0 으로 구분되고, 과도 구간 위상이 보인다
    """

    def __init__(self, variant: str, frequency: float,
                 stop_rate: float, eps: float = 1e-6):
        self.variant = variant
        self.phase = 0.0
        self.frequency = float(frequency)       # 명령 주파수 [Hz]
        self.stop_rate = float(stop_rate)       # v1s 전용: 위상 수렴 속도
        self.eps = eps

        self.effective_frequency = 0.0
        self.is_walking = False
        self.is_stopping = False

    def reset(self):
        """위상을 0 에서 다시 시작. 학습의 reset_phase_on_resample 과 동일."""
        self.phase = 0.0
        self.effective_frequency = 0.0
        self.is_walking = False
        self.is_stopping = False

    def update(self, dt: float, freq_cmd: float):
        """한 스텝 진행.

        Args:
            dt: 정책 주기 [s]
            freq_cmd: 이번 스텝의 주파수 명령 [Hz]. 0 이면 정지.
        """
        freq = max(0.0, float(freq_cmd))
        self.effective_frequency = freq
        self.is_walking = freq > self.eps

        if self.is_walking:
            # ── 보행: 정방향 증분 적분 (두 변형 공통) ──
            self.phase = math.fmod(self.phase + dt * freq, 1.0)
            if self.phase < 0.0:
                self.phase += 1.0
            self.is_stopping = False
            return

        if self.variant == "v1":
            # ── v1: 위상 동결 ──
            # 학습에서 freq=0 이면 phase += dt*0 이므로 값이 유지된다.
            # 0 으로 리셋하면 학습 분포와 어긋난다.
            self.is_stopping = False
            return

        # ── v0_1 / v0_2 / v1s: 0 을 향해 최단 경로로 선형 이동 ──
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
        """관측 성분. v1 은 2 차원, v1s 는 3 차원."""
        two_pi = 2.0 * math.pi
        c = math.cos(two_pi * self.phase)
        sn = math.sin(two_pi * self.phase)

        if self.variant == "v1":
            # 정지 시 (0, 0) 으로 게이팅.
            # cos^2 + sin^2 = 0 은 위상이 어떤 값이든 나올 수 없는 조합이라
            # 정책에게 "보행 중 아님" 을 명확히 알린다.
            gate = 1.0 if self.is_walking else 0.0
            return np.array([c * gate, sn * gate], dtype=np.float32)

        if self.variant == "v0_1":
            # 게이팅 없음, 주파수도 관측하지 않는다.
            # 정지 구분은 "위상이 얼어붙었는가" 를 히스토리가 담당한다.
            #   완전 정지 : [(1,0)] x 5   -> 5 프레임이 전부 같음
            #   보행 중   : 매 프레임 변화
            return np.array([c, sn], dtype=np.float32)

        # v0_2 / v1s: 게이팅 없음. 정지는 freq=0 으로 구분된다.
        return np.array([
            c, sn, self.effective_frequency / max(1e-6, frequency_scale),
        ], dtype=np.float32)


class ActorOnnxWalkingNode(Node):
    def __init__(self):
        super().__init__("actor_onnx_walking_node")

        # ── 기본 모델 경로 ──
        default_actor = "~/policy_v1s.onnx"
        if get_package_share_directory is not None:
            try:
                default_actor = os.path.join(
                    get_package_share_directory("humanoid_policy_runner"),
                    "models",
                    "policy_v1s.onnx",
                )
            except Exception:
                pass

        # 정책 변형. 관측 규격과 위상 처리가 달라진다.
        #   "v1"  : 관측 235 (프레임 47), 위상 동결 + cos/sin 게이팅
        #   "v1s" : 관측 240 (프레임 48), 위상 0 수렴 + 주파수 관측
        self.declare_parameter("policy_variant", "v1s")
        self.declare_parameter("actor_path", default_actor)
        self.declare_parameter("policy_rate_hz", 50.0)

        self.declare_parameter("cmd_vx", 0.0)
        self.declare_parameter("cmd_vy", 0.0)
        self.declare_parameter("cmd_wz", 0.0)

        self.declare_parameter("action_scale", 0.25)
        self.declare_parameter("clip_actions", 10.0)
        self.declare_parameter("base_ang_vel_scale", 0.2)
        self.declare_parameter("joint_vel_scale", 0.05)

        # ── 보행 위상 파라미터 (학습 cfg 와 반드시 일치) ──
        # velocity_v1s_env_cfg.py 의 StoppableGaitPhaseCommandCfg 참조.
        #   frequency_range=(1.8, 2.2) -> 이 범위 밖은 분포 외이므로
        #   gait_frequency 를 벗어난 값으로 두면 성능이 보장되지 않는다.
        self.declare_parameter("gait_frequency", 2.0)
        self.declare_parameter("gait_frequency_scale", 2.0)
        self.declare_parameter("gait_stop_rate", 2.0)
        self.declare_parameter("standing_threshold", 0.1)

        # gait_mode: "auto" | "march" | "stop" | "walk"
        #   auto  : 속도 0 이 stop_delay_s 이상 지속되면 완전 정지로 전환
        #   march : 속도 0 이어도 제자리 걷기 유지 (리듬 유지, 즉시 출발 가능)
        #   stop  : 속도와 무관하게 완전 정지 (주파수 0)
        #   walk  : 속도와 무관하게 주파수 유지 (디버그용)
        self.declare_parameter("gait_mode", "auto")
        self.declare_parameter("stop_delay_s", 1.0)

        self.declare_parameter("enable_action_filter", False)
        self.declare_parameter("tau_action_filter", 0.03)

        self.declare_parameter("max_target_delta_from_current_rad", 10.35)
        self.declare_parameter("max_target_abs_from_default_rad", 10.80)

        self.declare_parameter("startup_hold_seconds", 1.0)
        self.declare_parameter("use_default_pose_on_startup_hold", True)
        self.declare_parameter("resume_hold_seconds", 1.0)

        # ── 킥 핸드오프 파라미터 (2026-08-18 추가) ──
        # kick_yield_enable=False 면 /kick_trigger 를 무시한다(킥 러너를
        # 안 띄우는 구성에서 실수로 양보만 하고 아무도 안 움직이는 것을 방지).
        self.declare_parameter("kick_yield_enable", True)
        # 양보 워치독 [s]. walkkick_v1 이 죽거나 트리거가 게이트에서 거부되면
        # /kick_done 이 영영 오지 않는다. 그대로 두면 이 노드는 발행을 멈춘
        # 채 방치되어 로봇이 힘이 빠진다. 이 시간이 지나면 강제 복귀한다.
        # walkkick_v1 의 kick_duration_s(기본 3.0) + 접근/hold 여유를 감안해
        # 넉넉히 잡되, 너무 길면 방치 시간이 길어진다.
        # 양보 워치독 [s]. walkkick_v1 이 죽거나 트리거가 게이트에서 거부되면
        # /kick_done 이 영영 오지 않는다. 그대로 두면 이 노드는 발행을 멈춘
        # 채 방치되어 아무도 관절을 잡지 않는다. 이 시간이 지나면 강제 복귀한다.
        #
        # ★ 자동 계산 (-1.0) 을 권장한다 ★
        #   킥 총 소요는 kick_duration_s 하나로 결정된다(트리거 직후의
        #   default 자세 hold 도 그 안에 포함된다 — walkkick 러너가
        #   kick_start_time 과 resume_time 을 동시에 세우고 kick_start_time
        #   기준으로 종료를 판정하기 때문). 따라서
        #       워치독 = kick_duration_s + kick_yield_margin_s
        #   로 두면 kick_duration 을 바꿔도 자동으로 따라간다. 고정값으로
        #   두면 kick_duration 만 늘렸을 때 워치독이 먼저 터져 킥이 중간에
        #   끊긴다(실제로 겪기 쉬운 실수).
        #
        # ★ 순서 불변식 ★
        #   이 워치독 < visual_servo 의 kick_wait_timeout_s 여야 한다.
        #   서보가 먼저 포기하면 서보는 DONE 으로 가면서 /kick_done 을
        #   발행하지 않는데, FSM 의 ALIGN_KICK 은 /kick_done 으로만
        #   빠져나오는 하드 래치라 교착에 빠진다. 이 노드가 먼저 복귀해
        #   /kick_done(복구 신호)을 쏘면 셋이 한 번에 정상화된다.
        #
        #   -1.0  : 자동 = kick_duration_s + kick_yield_margin_s
        #   0 이하(-1.0 제외) : 워치독 비활성(권장하지 않음)
        #   양수  : 그 값을 그대로 사용
        self.declare_parameter("kick_yield_timeout_s", -1.0)
        # 자동 계산 시 kick_duration_s 에 더할 여유 [s].
        self.declare_parameter("kick_yield_margin_s", 1.0)
        # 자동 계산의 기준이 되는 킥 길이 [s]. walkkick_v1 러너의
        # kick_duration_s 와 같은 값을 넣어야 한다(launch 가 함께 넘긴다).
        # 노드가 분리되어 있어 직접 읽을 수 없으므로 launch 가 짝을 맞춘다.
        self.declare_parameter("kick_duration_hint_s", 3.0)
        # visual_servo 의 kick_wait_timeout_s 값(참조용).
        #   이 노드가 직접 읽을 수 없으므로 launch 가 같은 값을 넘겨준다.
        #   워치독이 이 값보다 크면 순서 불변식이 깨지므로 기동 시 경고한다.
        #   0 이하면 검사하지 않는다.
        self.declare_parameter("servo_kick_wait_timeout_ref_s", 0.0)
        # 킥 복귀 전용 hold [s]. -1.0 이면 resume_hold_seconds 를 따른다.
        # ★ 낙상 복귀와 분리하는 이유 ★
        #   낙상 복귀는 로봇이 바닥에서 일어난 직후라 충분한 안정화가
        #   필요하지만(1.0s), 킥 복귀는 서 있는 상태에서 자세만 흐트러진
        #   것이라 더 짧아도 된다. 두 값을 하나로 묶으면 킥 응답성을
        #   위해 낙상 안전 마진까지 깎게 된다.
        #   0.0 = hold 없이 즉시 보행 (테스트용, 아래 주의사항 참조)
        self.declare_parameter("kick_resume_hold_seconds", -1.0)

        self.declare_parameter("default_joint_pos", [0.0] * 12)
        self.declare_parameter("debug_print_every", 50)
        self.declare_parameter("pelvis_pitch_offset", 0.0)

        # ── 가상 중력 오프셋 ──
        # 정책이 보는 projected_gravity 를 트림한다.
        #   +x = "앞으로 숙였다고 인지" -> 정책이 뒤로 보정 -> 실제로는 더 세워짐
        #   +y = "왼쪽으로 기울었다고 인지" -> 오른쪽으로 보정
        #
        # ★ IMU 장착 오차 보정이 아니다 ★
        # 장착 오차는 imu_parser 의 imu_mount_roll/pitch_deg 가 담당한다.
        # 여기는 "정책이 학습한 자세와 실기의 정적/동적 편차" 를 메우는
        # 트림이다. 둘을 섞으면 낙상 감지(fall_manager)의 기울기 임계값이
        # 어긋나므로 반드시 분리해서 쓸 것.
        #
        # 튜닝 예시 (v0_2, 제자리→전진):
        #   정적 편차  실기 +2.69도 vs sim +1.96도  = 0.73도  (허용 범위)
        #   보행 편차  실기 +3.78도 vs sim +1.50도  = 2.28도
        #   동적 성분 1.55도 -> sin(1.55도) = 0.027 이 시작값
        #
        # 실시간 튜닝:
        #   ros2 param set /actor_onnx_walking_node grav_offset_x 0.027
        self.declare_parameter("grav_offset_x", 0.0)
        self.declare_parameter("grav_offset_y", 0.0)
        self.declare_parameter("grav_offset_z", 0.0)

        self.declare_parameter("onnx_provider", "auto")

        # ── IMU 보정 ─────────────────────────────────────────────
        # gravity_offset: 관측에 넣기 전 projected_gravity 에서 뺄 값.
        #   IMU 장착각 오차나 프레임 변형 때문에, 로봇이 물리적으로
        #   똑바로 서 있어도 측정 중력벡터가 (0,0,-1) 이 아닐 수 있다.
        #   이 편차는 정책 입장에서 "계속 기울어져 있다" 는 신호라
        #   한쪽으로 보정하려 들고, 결과적으로 보행이 한쪽으로 쏠린다.
        # gyro_bias: base_ang_vel 에서 뺄 값 [rad/s]. 스케일 적용 전 원단위.
        self.declare_parameter("gravity_offset", [0.0, 0.0, 0.0])
        self.declare_parameter("gyro_bias", [0.0, 0.0, 0.0])

        # startup_hold 동안 정지 상태 샘플을 모아 위 두 값을 자동 산출한다.
        # ★ 전제: hold 구간에 로봇이 평평한 바닥에 default 자세로 가만히
        #   서 있어야 한다 ★ 매달아 놓았거나 움직이는 중이면 엉뚱한 값이
        #   나오므로, 반드시 결과 로그를 확인하고 이상하면 재실행할 것.
        # 산출된 값은 로그로 찍히므로, 안정되면 파라미터에 고정하고
        # calibrate_on_startup 을 False 로 두는 것을 권장한다.
        self.declare_parameter("calibrate_on_startup", False)
        self.declare_parameter("calib_min_samples", 25)
        # 샘플 분산이 이 값을 넘으면 "가만히 있지 않았다" 로 보고 보정을 버린다.
        self.declare_parameter("calib_max_gyro_std", 0.05)   # [rad/s]
        self.declare_parameter("calib_max_grav_std", 0.02)   # [-]

        self.declare_parameter("enable_csv_log", True)
        self.declare_parameter("csv_output_dir", "~/policy_logs")
        self.declare_parameter("csv_flush_every", 25)

        # ── 정책 변형 결정 ──
        self.variant = str(self.get_parameter("policy_variant").value).strip().lower()
        if self.variant not in VARIANT_SPEC:
            raise RuntimeError(
                f"policy_variant 는 {list(VARIANT_SPEC)} 중 하나여야 합니다: {self.variant}")
        self.gait_dim = VARIANT_SPEC[self.variant]
        self.frame_slices, self.frame_dim = _make_frame_spec(self.gait_dim)
        self.obs_dim = self.frame_dim * HISTORY_LEN

        # ── 파라미터 읽기 ──
        # 빈 문자열이 넘어오면 os.path.expanduser("") 가 홈 디렉터리로 해석되어
        # "actor_path not found: /home/user" 같은 혼란스러운 에러가 난다.
        # launch 에서 인자를 안 넘긴 경우를 대비해 기본 경로로 되돌린다.
        _actor_param = str(self.get_parameter("actor_path").value).strip()
        if not _actor_param:
            _actor_param = default_actor
            self.get_logger().warn(
                f"actor_path 가 비어 있어 기본 경로를 사용합니다: {default_actor}")
        self.actor_path = _expand_path(_actor_param)
        self.policy_rate_hz = float(self.get_parameter("policy_rate_hz").value)
        self.policy_dt = 1.0 / max(1e-6, self.policy_rate_hz)

        self.cmd = np.array([
            float(self.get_parameter("cmd_vx").value),
            float(self.get_parameter("cmd_vy").value),
            float(self.get_parameter("cmd_wz").value),
        ], dtype=np.float32)

        self.action_scale = float(self.get_parameter("action_scale").value)
        self.clip_actions = float(self.get_parameter("clip_actions").value)
        self.base_ang_vel_scale = float(self.get_parameter("base_ang_vel_scale").value)
        self.joint_vel_scale = float(self.get_parameter("joint_vel_scale").value)

        self.gait_frequency = float(self.get_parameter("gait_frequency").value)
        self.gait_frequency_scale = float(self.get_parameter("gait_frequency_scale").value)
        self.gait_stop_rate = float(self.get_parameter("gait_stop_rate").value)
        self.standing_threshold = float(self.get_parameter("standing_threshold").value)
        self.gait_mode = str(self.get_parameter("gait_mode").value).lower()
        self.stop_delay_s = float(self.get_parameter("stop_delay_s").value)

        self.enable_action_filter = bool(self.get_parameter("enable_action_filter").value)
        self.tau_action_filter = float(self.get_parameter("tau_action_filter").value)
        self.alpha = float(np.exp(-self.policy_dt / max(1e-6, self.tau_action_filter)))

        self.max_target_delta_from_current = float(
            self.get_parameter("max_target_delta_from_current_rad").value)
        self.max_target_abs_from_default = float(
            self.get_parameter("max_target_abs_from_default_rad").value)

        self.startup_hold_seconds = float(self.get_parameter("startup_hold_seconds").value)
        self.use_default_pose_on_startup_hold = bool(
            self.get_parameter("use_default_pose_on_startup_hold").value)
        self.resume_hold_seconds = float(self.get_parameter("resume_hold_seconds").value)

        self.debug_print_every = int(self.get_parameter("debug_print_every").value)
        self.pelvis_pitch_offset = float(self.get_parameter("pelvis_pitch_offset").value)

        self.grav_offset = np.array([
            float(self.get_parameter("grav_offset_x").value),
            float(self.get_parameter("grav_offset_y").value),
            float(self.get_parameter("grav_offset_z").value),
        ], dtype=np.float32)

        self.gravity_offset = np.array(
            self.get_parameter("gravity_offset").value, dtype=np.float32).reshape(-1)
        self.gyro_bias = np.array(
            self.get_parameter("gyro_bias").value, dtype=np.float32).reshape(-1)
        if self.gravity_offset.size != 3:
            self.gravity_offset = np.zeros(3, dtype=np.float32)
        if self.gyro_bias.size != 3:
            self.gyro_bias = np.zeros(3, dtype=np.float32)

        self.calibrate_on_startup = bool(self.get_parameter("calibrate_on_startup").value)
        self.calib_min_samples = int(self.get_parameter("calib_min_samples").value)
        self.calib_max_gyro_std = float(self.get_parameter("calib_max_gyro_std").value)
        self.calib_max_grav_std = float(self.get_parameter("calib_max_grav_std").value)

        # ── 파라미터 동적 변경 콜백 ──
        # 이걸 등록하지 않으면 __init__ 에서 캐시한 값이 고정되어
        # `ros2 param set` 이 동작하지 않는다.
        self.add_on_set_parameters_callback(self._on_set_parameters)

        # ── ONNX 모델 로드 ──
        if not os.path.isfile(self.actor_path):
            raise RuntimeError(f"actor_path not found: {self.actor_path}")

        providers = self._select_providers(str(self.get_parameter("onnx_provider").value).lower())
        self.session = ort.InferenceSession(self.actor_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        in_shape = self.session.get_inputs()[0].shape
        self.get_logger().info(f"ONNX loaded: {self.actor_path}")
        self.get_logger().info(f"ONNX provider: {self.session.get_providers()[0]}")
        self.get_logger().info(f"  input:  {self.input_name} {in_shape}")
        out_shape = self.session.get_outputs()[0].shape
        self.get_logger().info(f"  output: {self.output_name} {out_shape}")

        # 관측 차원 검증. v0 정책(225)을 잘못 물리면 여기서 걸린다.
        try:
            model_dim = int(in_shape[-1])
            if model_dim != self.obs_dim:
                # 변형을 잘못 지정한 경우가 대부분이라, 다른 변형이 맞는지
                # 찾아서 알려준다.
                hint = ""
                for name, gd in VARIANT_SPEC.items():
                    _, fd = _make_frame_spec(gd)
                    if fd * HISTORY_LEN == model_dim:
                        hint = f"  → policy_variant:={name} 로 실행하십시오.\n"
                raise RuntimeError(
                    f"ONNX 입력 차원 불일치: 모델={model_dim}, "
                    f"기대={self.obs_dim} (policy_variant={self.variant}).\n"
                    + hint
                    + "  v0 정책(225)은 이 노드로 실행할 수 없습니다."
                )
        except (TypeError, ValueError):
            self.get_logger().warn("ONNX 입력 차원을 확인할 수 없습니다(동적 shape).")

        # ── 기본 자세 ──
        isaac_default = _load_default_joint_pos()
        param_default = np.array(
            self.get_parameter("default_joint_pos").value, dtype=np.float32).reshape(-1)

        if len(isaac_default) > 0:
            vals = [float(isaac_default.get(jn, 0.0)) for jn in POLICY_JOINTS]
            self.default_q12 = np.array(vals, dtype=np.float32)
            self.get_logger().info("Loaded default_joint_pos from robot_init_behavior.")
        elif param_default.size == 12:
            self.default_q12 = param_default.astype(np.float32)
            self.get_logger().warn("Using default_joint_pos from ROS parameter.")
        else:
            self.default_q12 = np.zeros((12,), dtype=np.float32)
            self.get_logger().warn("default_joint_pos fallback to zeros.")

        self.default_qd12 = np.zeros((12,), dtype=np.float32)

        # ── 상태 변수 ──
        self.last_joint_state: Optional[JointState] = None
        self.last_imu: Optional[Imu] = None
        self.last_gravity: Optional[Vector3Stamped] = None

        self.q12 = np.zeros((12,), dtype=np.float32)
        self.qd12 = np.zeros((12,), dtype=np.float32)

        self.last_action12 = np.zeros((12,), dtype=np.float32)
        self.filt_action12 = np.zeros((12,), dtype=np.float32)

        self.history = FrameHistory(H=HISTORY_LEN, dim=self.frame_dim)
        self.history_initialized = False

        # ── 보행 위상 적분기 ──
        self.gait = GaitPhaseIntegrator(
            variant=self.variant,
            frequency=self.gait_frequency,
            stop_rate=self.gait_stop_rate,
        )
        # 속도 명령이 0 이 된 시각. auto 모드에서 완전 정지 전환 판정에 쓴다.
        self._standing_since: Optional[float] = None

        self.start_wall_time = time.time()
        self.tick_count = 0

        # ── CSV ──
        self.enable_csv_log = bool(self.get_parameter("enable_csv_log").value)
        self.csv_flush_every = int(self.get_parameter("csv_flush_every").value)
        self._csv_file = None
        self._csv_writer = None
        self._csv_row_count = 0
        self._csv_closed = False
        if self.enable_csv_log:
            self._init_csv_logger()

        # ── [2026-08 CPU 최적화] projected gravity 취득 방식 ──────────────
        #   derive    : /imu/data.orientation 에서 파생 계산 (구독 없음, 기본)
        #   subscribe : 기존대로 /imu/projected_gravity 구독 (롤백 경로)
        #   verify    : 둘 다 받아 비교 로그 (실기 대조용, CPU 절감 없음)
        #
        #   오프라인 검증에서 랜덤 쿼터니언 20만 개에 대해 파생값이 원본과
        #   비트 단위로 일치함을 확인했다(최대 오차 0.0). 그래도 실기에서
        #   한 번 대조하고 넘어가고 싶으면 verify 로 띄울 것:
        #     ros2 run ... --ros-args -p gravity_source:=verify
        self.declare_parameter("gravity_source", "derive")
        self.gravity_source = str(
            self.get_parameter("gravity_source").value).lower()
        if self.gravity_source not in ("derive", "subscribe", "verify"):
            self.get_logger().warn(
                f"gravity_source='{self.gravity_source}' 는 알 수 없는 값 — "
                f"'derive' 로 강제합니다.")
            self.gravity_source = "derive"
        # verify 모드용: 파생값과 구독값의 최대 절대차 누적
        self._grav_verify_max = 0.0
        self._grav_verify_n = 0
        self._grav_verify_last_log = 0.0
        self.get_logger().info(f"gravity_source = {self.gravity_source}")

        # ── ROS 구독/발행 ──
        self.create_subscription(JointState, "/joint_states", self.cb_joint, 20)
        self.create_subscription(Imu, "/imu/data", self.cb_imu, 20)

        # ── [2026-08 CPU 최적화] projected gravity 구독 제거 ────────────────
        #   /imu/projected_gravity 는 /imu/data.orientation 만의 순수 함수라
        #   정보량이 0 인 중복 토픽이다(imu_math.projected_gravity_from_quat).
        #   두 토픽을 다 구독하면 executor 가 초당 400번 깨어나는데, rclpy 는
        #   깨어날 때마다 wait set 을 Python 에서 재구축하므로 그게 그대로 CPU 다.
        #   파생 계산으로 바꾸면 깨우기가 200 으로 절반이 된다.
        #
        #   ★ /imu/data 는 200Hz 그대로 유지한다 ★
        #     이 노드의 관측 신선도는 전혀 바뀌지 않는다. 학습 시 설정한
        #     action delay 와의 정합성이 깨지지 않으므로 sim-to-real 리스크 없음.
        #     (데시메이션은 fall_manager/walk_odometry 같은 50Hz 소비자에게만 적용)
        #
        #   gravity_source 로 동작을 고를 수 있다:
        #     "derive"    (기본) 파생 계산. 구독 안 함.
        #     "subscribe" 기존 동작. 문제 발생 시 즉시 되돌리는 경로.
        #     "verify"    둘 다 받아 차이를 주기적으로 로그. 실기 대조용.
        #                 (이 모드는 깨우기가 줄지 않는다. 검증 전용)
        if self.gravity_source in ("subscribe", "verify"):
            self.create_subscription(
                Vector3Stamped, "/imu/projected_gravity", self.cb_gravity, 20)
        self.create_subscription(Twist, "/cmd_vel", self.cb_cmd_vel, 10)
        self.create_subscription(String, "/robot_state", self.cb_robot_state, 10)

        # ── 킥 핸드오프 (2026-08-18 추가) ──
        # 이 노드와 actor_onnx_walkkick_v1_runner 는 둘 다 /joint_goals 에
        # 발행한다. 동시에 쓰면 매 틱 서로의 목표를 덮어써서 즉시 발산하므로,
        # 어느 순간에도 발행자가 하나뿐이도록 소유권을 넘긴다.
        #
        #   [보행] --/kick_trigger--> [양보: 발행 완전 중단]
        #                                   |
        #                          [walkkick_v1 접근+킥]
        #                                   |
        #          <--/kick_done------------+
        #   [복귀: history 리셋 + resume_hold ease-in]
        #
        # 양보 중에는 default 자세조차 발행하지 않는다. 한 틱이라도 겹치면
        # 두 정책의 목표가 번갈아 나가 관절이 진동한다.
        self.create_subscription(Bool, "/kick_trigger", self.cb_kick_trigger, 10)
        self.create_subscription(Bool, "/kick_done", self.cb_kick_done, 10)
        # 워치독 복구 신호용 퍼블리셔.
        #   워치독이 터졌다는 건 킥 노드가 /kick_done 을 못 낸 상황이다.
        #   이 노드만 조용히 복귀하면 visual_servo 는 KICK_WAIT 에서,
        #   strategy_fsm 은 ALIGN_KICK 하드 래치에서 계속 갇힌다.
        #   복구 신호를 대신 쏴서 셋이 함께 정상 상태로 돌아가게 한다.
        #   ※ 의미상 "킥 완료"는 아니지만, FSM 이 /kick_done 을 받고 하는 일
        #     (post_kick_settle 후 재접근 재평가)이 곧 원하는 복구 동작이다.
        self.kick_done_pub = self.create_publisher(Bool, "/kick_done", 10)

        # 상위 계층(FSM) 또는 조이패드가 보행 모드/주파수를 지정하는 토픽.
        # joypad_teleop_v1s_node 는 TRANSIENT_LOCAL(latched)로 발행하므로,
        # 이 노드가 나중에 떠도 마지막 값을 즉시 받는다. 구독 쪽 durability 를
        # 맞추지 않으면 latch 된 값을 못 받고 다음 변경까지 기본값으로 돈다.
        latched_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        # "auto" | "march" | "stop" | "walk"
        self.create_subscription(String, "/gait_mode", self.cb_gait_mode, latched_qos)
        # 보행 주파수 [Hz]. 인조잔디 등에서 실시간 튜닝용.
        self.create_subscription(
            Float32, "/gait_frequency", self.cb_gait_frequency, latched_qos)

        self.goal_pub = self.create_publisher(JointState, "/joint_goals", 10)
        # RViz / Foxglove 에서 위상을 직접 확인하기 위한 디버그 토픽
        self.phase_pub = self.create_publisher(Float32, "/gait/phase", 10)
        self.freq_pub = self.create_publisher(Float32, "/gait/effective_frequency", 10)
        self.state_pub = self.create_publisher(String, "/gait/state", 10)

        self.robot_state = "NORMAL"
        self.resume_time = None
        # 이번 복귀에 적용할 hold 길이 [s]. 낙상 복귀와 킥 복귀가 서로
        # 다른 값을 쓸 수 있으므로, resume_time 을 세울 때 함께 정한다.
        self._resume_hold_dur = self.resume_hold_seconds

        # ── 킥 핸드오프 상태 ──
        self.kick_yield_enable = bool(
            self.get_parameter("kick_yield_enable").value)
        _kyt = float(self.get_parameter("kick_yield_timeout_s").value)
        _kym = float(self.get_parameter("kick_yield_margin_s").value)
        _kdh = float(self.get_parameter("kick_duration_hint_s").value)
        if _kyt < 0.0:
            # 자동: 킥 길이 + 여유. kick_duration 을 바꿔도 따라간다.
            self.kick_yield_timeout_s = max(0.5, _kdh + _kym)
            self._kick_yield_timeout_src = (
                f"자동(kick_duration {_kdh:.1f} + margin {_kym:.1f})")
        else:
            self.kick_yield_timeout_s = _kyt
            self._kick_yield_timeout_src = "고정값"
        # ── 순서 불변식 검사 ──
        #   워치독 < visual_servo.kick_wait_timeout_s 여야 한다.
        #   깨지면 서보가 먼저 DONE 으로 빠지는데 /kick_done 은 발행되지 않아
        #   FSM 의 ALIGN_KICK 하드 래치가 안 풀리고 교착된다. 조용히 넘기면
        #   경기 중에야 드러나므로 기동 시점에 크게 경고한다.
        _servo_ref = float(
            self.get_parameter("servo_kick_wait_timeout_ref_s").value)
        if _servo_ref > 0.0 and self.kick_yield_timeout_s >= _servo_ref:
            self.get_logger().error(
                f"[kick] ★순서 불변식 위반★ 양보 워치독"
                f"({self.kick_yield_timeout_s:.1f}s) >= visual_servo 의 "
                f"kick_wait_timeout_s({_servo_ref:.1f}s). 킥 실패 시 서보가 "
                "먼저 포기해 FSM 의 ALIGN_KICK 래치가 풀리지 않고 교착됩니다. "
                "vision_servo.yaml 의 kick_wait_timeout_s 를 "
                f"{self.kick_yield_timeout_s + 1.0:.1f} 이상으로 올리거나 "
                "kick_duration 을 줄이세요.")
        _krh = float(self.get_parameter("kick_resume_hold_seconds").value)
        # 음수면 "지정 안 함" → 일반 resume_hold_seconds 를 그대로 쓴다.
        self.kick_resume_hold_seconds = (
            self.resume_hold_seconds if _krh < 0.0 else _krh)
        # True 이면 /joint_goals 발행을 완전히 중단하고 킥 러너에 소유권을 넘긴 상태.
        self._kick_yield = False
        # 양보 시작 시각(워치독 판정용).
        self._kick_yield_since = None

        self.get_logger().info(
            f"variant={self.variant} policy_rate_hz={self.policy_rate_hz} "
            f"(event-driven) frame={self.frame_dim} obs_dim={self.obs_dim}")
        self.get_logger().info(
            f"gait: freq={self.gait_frequency}Hz scale={self.gait_frequency_scale} "
            f"stop_rate={self.gait_stop_rate} mode={self.gait_mode} "
            f"(최대 정지 시간 {0.5 / max(1e-6, self.gait_stop_rate):.3f}s)")
        self.get_logger().info(f"action_scale={self.action_scale}")
        if np.any(self.grav_offset != 0.0):
            deg = np.degrees(np.arcsin(np.clip(self.grav_offset[:2], -1.0, 1.0)))
            self.get_logger().info(
                f"grav_offset=({self.grav_offset[0]:+.4f}, {self.grav_offset[1]:+.4f}, "
                f"{self.grav_offset[2]:+.4f})  = pitch{deg[0]:+.2f}deg roll{deg[1]:+.2f}deg")

    # ──────────────────────────────────────
    # 파라미터 동적 변경
    # ──────────────────────────────────────
    def _on_set_parameters(self, params):
        from rcl_interfaces.msg import SetParametersResult

        for p in params:
            if p.name == "gait_frequency":
                self.gait_frequency = float(p.value)
            elif p.name == "gait_stop_rate":
                self.gait_stop_rate = float(p.value)
                self.gait.stop_rate = self.gait_stop_rate
            elif p.name == "gait_mode":
                self.gait_mode = str(p.value).lower()
            elif p.name == "stop_delay_s":
                self.stop_delay_s = float(p.value)
            elif p.name == "standing_threshold":
                self.standing_threshold = float(p.value)
            elif p.name == "action_scale":
                self.action_scale = float(p.value)
            elif p.name == "pelvis_pitch_offset":
                self.pelvis_pitch_offset = float(p.value)
            elif p.name == "grav_offset_x":
                self.grav_offset[0] = float(p.value)
            elif p.name == "grav_offset_y":
                self.grav_offset[1] = float(p.value)
            elif p.name == "grav_offset_z":
                self.grav_offset[2] = float(p.value)
            elif p.name == "debug_print_every":
                self.debug_print_every = int(p.value)

        return SetParametersResult(successful=True)

    # ──────────────────────────────────────
    # ONNX provider
    # ──────────────────────────────────────
    def _select_providers(self, mode: str) -> list:
        available = ort.get_available_providers()
        self.get_logger().info(f"ONNX available providers: {available}")

        if mode == "cpu":
            return ["CPUExecutionProvider"]
        elif mode == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif mode == "tensorrt":
            return ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = []
            if "TensorrtExecutionProvider" in available:
                providers.append("TensorrtExecutionProvider")
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")
            return providers

    # ──────────────────────────────────────
    # 콜백
    # ──────────────────────────────────────
    def cb_joint(self, msg: JointState):
        self.last_joint_state = msg
        # /joint_states 에는 body_motor(다리+팔)와 neck_state_parser(목 2관절)가
        # 함께 발행된다. 다리 12관절이 든 메시지일 때만 정책을 1스텝 트리거한다.
        if not POLICY_JOINTS_SET.issubset(msg.name):
            return
        self.control_tick()

    def cb_imu(self, msg: Imu):
        self.last_imu = msg

        # ── [2026-08 CPU 최적화] projected gravity 파생 ────────────────────
        #   구독 대신 여기서 만든다. 값은 imu_parser 가 보내주던 것과
        #   비트 단위로 동일하다(imu_math 주석의 동기화 의무 참조).
        #   subscribe 모드에서는 cb_gravity 가 채우므로 건드리지 않는다.
        if self.gravity_source == "subscribe":
            return

        gx, gy, gz = projected_gravity_from_quat(msg.orientation)

        # Vector3Stamped 객체를 200Hz 로 새로 만들면 그것도 비용이므로
        # 최초 1회만 할당하고 이후에는 필드만 갱신한다.
        #   - 하위 코드가 self.last_gravity.vector.x 형태로 읽으므로 타입 유지.
        #   - SingleThreadedExecutor 라 콜백과 소비가 같은 스레드에서 순차
        #     실행된다 → in-place 갱신이 안전하다.
        #   - "첫 IMU 수신 전에는 None" 이라는 기존 준비상태 판정
        #     (`self.last_gravity is not None`)을 그대로 보존하려고
        #     미리 할당하지 않는다.
        if self.last_gravity is None:
            self.last_gravity = Vector3Stamped()
            self.last_gravity.header.frame_id = "imu_link"
        self.last_gravity.header.stamp = msg.header.stamp
        self.last_gravity.vector.x = gx
        self.last_gravity.vector.y = gy
        self.last_gravity.vector.z = gz

    def cb_gravity(self, msg: Vector3Stamped):
        """/imu/projected_gravity 콜백.

        gravity_source 가 "subscribe" 또는 "verify" 일 때만 구독이 걸린다.
        기본값 "derive" 에서는 호출되지 않는다.
        """
        if self.gravity_source == "verify":
            # 파생값(cb_imu 가 채운 것)과 구독값을 대조한다.
            if self.last_gravity is not None:
                d = max(
                    abs(self.last_gravity.vector.x - msg.vector.x),
                    abs(self.last_gravity.vector.y - msg.vector.y),
                    abs(self.last_gravity.vector.z - msg.vector.z),
                )
                if d > self._grav_verify_max:
                    self._grav_verify_max = d
                self._grav_verify_n += 1
                now = time.time()
                if now - self._grav_verify_last_log >= 5.0:
                    self._grav_verify_last_log = now
                    self.get_logger().info(
                        f"[grav-verify] n={self._grav_verify_n} "
                        f"max|derive-subscribe|={self._grav_verify_max:.3e} "
                        f"(1e-6 이하면 동일한 것으로 간주)")
            return

        self.last_gravity = msg

    def cb_cmd_vel(self, msg: Twist):
        self.cmd[0] = float(msg.linear.x)
        self.cmd[1] = float(msg.linear.y)
        self.cmd[2] = float(msg.angular.z)

    def cb_gait_mode(self, msg: String):
        if self.variant in ("v0_1", "v1"):
            if not getattr(self, "_warned_mode_v1", False):
                self.get_logger().warn(
                    f"[gait_mode] {self.variant} 정책은 주파수가 속도 명령으로 "
                    "결정되므로 "
                    "보행 모드를 무시합니다. (제자리 걷기/정지 구분 없음)")
                self._warned_mode_v1 = True
            return
        mode = msg.data.strip().lower()
        if mode not in ("auto", "march", "stop", "walk"):
            self.get_logger().warn(f"[gait_mode] 알 수 없는 모드: {mode} (무시)")
            return
        if mode != self.gait_mode:
            self.get_logger().info(f"[gait_mode] {self.gait_mode} -> {mode}")
        self.gait_mode = mode

    def cb_gait_frequency(self, msg: Float32):
        f = float(msg.data)
        # 학습 범위(1.8~2.2Hz) 밖은 분포 외다. 막지는 않되 경고한다.
        if f > 1e-6 and not (1.5 <= f <= 2.6):
            self.get_logger().warn(
                f"[gait_frequency] {f:.2f}Hz 는 학습 분포(1.8~2.2) 밖입니다. "
                f"성능이 보장되지 않습니다.")
        self.gait_frequency = max(0.0, f)

    # ── 킥 핸드오프 (2026-08-18 추가) ────────────────────────────
    def cb_kick_trigger(self, msg: Bool):
        """킥 시작 → /joint_goals 소유권을 walkkick_v1 러너에 넘긴다."""
        if not msg.data:
            return
        if not self.kick_yield_enable:
            self.get_logger().warn(
                "[kick] /kick_trigger 수신했으나 kick_yield_enable=False — 무시")
            return
        if self._kick_yield:
            return          # 이미 양보 중(중복 트리거)
        self._kick_yield = True
        self._kick_yield_since = time.time()
        self.get_logger().info(
            "[kick] 양보 시작 — /joint_goals 발행 중단. "
            f"walkkick_v1 인계 (워치독 {self.kick_yield_timeout_s:.1f}s "
            f"— {self._kick_yield_timeout_src})")

    def cb_kick_done(self, msg: Bool):
        """킥 종료 → 보행 소유권 회수."""
        if not msg.data:
            return
        if not self._kick_yield:
            return          # 양보 중이 아니면 남의 신호(무시)
        self._resume_from_kick("kick_done 수신")

    def _resume_from_kick(self, reason: str):
        """양보를 풀고 보행으로 복귀한다.

        킥 러너가 관절을 완전히 다른 자세로 두고 끝나므로, 낙상 복귀
        (cb_robot_state) 와 똑같이 히스토리·last_action·위상을 전부 리셋한다.
        리셋 없이 복귀하면 킥 직전 액션이 obs 에 섞여 첫 프레임부터 발산한다.
        """
        self._kick_yield = False
        self._kick_yield_since = None
        self.history_initialized = False
        self.resume_time = time.time()
        self._resume_hold_dur = self.kick_resume_hold_seconds
        self.last_action12 = np.zeros((12,), dtype=np.float32)
        self.gait.reset()
        self._standing_since = None
        if self._resume_hold_dur <= 0.0:
            self.get_logger().info(
                f"[kick] 보행 복귀 ({reason}) — hold 없음(즉시 보행)")
        else:
            self.get_logger().info(
                f"[kick] 보행 복귀 ({reason}) — resume_hold "
                f"{self._resume_hold_dur:.2f}s ease-in")

    def cb_robot_state(self, msg: String):
        new_state = msg.data
        if new_state != self.robot_state:
            self.get_logger().info(f"[robot_state] {self.robot_state} -> {new_state}")
            if new_state != "NORMAL":
                self.history_initialized = False
            else:
                self.resume_time = time.time()
                # 낙상 복귀는 킥 복귀와 달리 바닥에서 일어난 직후라
                # 안정화 시간이 더 필요하다. 항상 일반 resume_hold 를 쓴다.
                self._resume_hold_dur = self.resume_hold_seconds
                # history 오염 방지: 넘어지기 직전 액션이 남아 있으면
                # 복귀 후 첫 프레임에 섞여 들어가 정책이 발산한 상태로
                # 시작한다.
                self.last_action12 = np.zeros((12,), dtype=np.float32)
                # 위상도 초기화. 학습의 reset_phase_on_resample 과 동일하게
                # 항상 위상 0(양발 stance)에서 재개한다.
                self.gait.reset()
                self._standing_since = None
        self.robot_state = new_state

    def policy_enabled(self) -> bool:
        return self.robot_state == "NORMAL"

    def ready(self) -> bool:
        return (
            self.last_joint_state is not None and
            self.last_imu is not None and
            self.last_gravity is not None
        )

    # ──────────────────────────────────────
    # 센서 처리
    # ──────────────────────────────────────
    def extract_q_qd_from_joint_state(self):
        js = self.last_joint_state
        assert js is not None

        name_to_idx = {name: i for i, name in enumerate(js.name)}
        missing = [jn for jn in POLICY_JOINTS if jn not in name_to_idx]
        if missing:
            raise RuntimeError(f"joint_states missing joints: {missing}")

        q = np.zeros((12,), dtype=np.float32)
        qd = np.zeros((12,), dtype=np.float32)

        for i, jn in enumerate(POLICY_JOINTS):
            idx = name_to_idx[jn]
            if idx >= len(js.position):
                raise RuntimeError(f"joint_states position missing for {jn}")
            q[i] = float(js.position[idx])
            qd[i] = float(js.velocity[idx]) if idx < len(js.velocity) else 0.0

        return q, qd

    # ──────────────────────────────────────
    # 보행 주파수 결정
    # ──────────────────────────────────────
    def resolve_gait_frequency(self, vel_cmd: np.ndarray, in_hold: bool) -> float:
        """이번 스텝에 적용할 주파수를 결정한다.

        학습(StoppableGaitPhaseCommand._update_command)의 판정 로직을
        실기 상황에 맞게 옮긴 것이다.

        학습에서는 "속도 0 인 env 중 rel_stop_when_standing 확률로 완전 정지"
        였지만, 실기에서는 확률이 아니라 상위 계층의 의도로 결정된다.
        gait_mode 파라미터/토픽이 그 역할을 한다.
        """
        # hold 구간: 학습의 freeze_during_reset_hold 와 동일하게 위상 정지
        if in_hold:
            return 0.0

        speed = float(np.linalg.norm(vel_cmd[:3]))
        standing = speed < self.standing_threshold

        # ── v0_1 / v1: 주파수는 전적으로 속도 명령이 결정한다 ──
        # 학습의 GaitPhaseCommand 가 standing 이면 freq 를 0 으로 강제
        # 게이팅했으므로, gait_mode 개념 자체가 없다. 조이패드가 모드를
        # 보내와도 무시한다 (경고는 cb_gait_mode 에서 한 번만).
        if self.variant in ("v0_1", "v1"):
            return 0.0 if standing else self.gait_frequency

        # 속도 0 지속 시간 추적 (auto 모드용)
        now = time.time()
        if standing:
            if self._standing_since is None:
                self._standing_since = now
        else:
            self._standing_since = None

        if self.gait_mode == "stop":
            return 0.0
        if self.gait_mode == "walk":
            return self.gait_frequency
        if self.gait_mode == "march":
            # 속도 0 이어도 리듬 유지
            return self.gait_frequency

        # auto: 속도 0 이 stop_delay_s 이상 지속되면 완전 정지.
        # 그 전까지는 제자리 걷기로 리듬을 유지해 즉시 출발할 수 있게 한다.
        if standing and self._standing_since is not None:
            if (now - self._standing_since) >= self.stop_delay_s:
                return 0.0
        return self.gait_frequency

    # ──────────────────────────────────────
    # 프레임 구성
    # ──────────────────────────────────────
    def one_frame(self, vel_cmd: np.ndarray) -> np.ndarray:
        """관측 1 프레임 생성 (v1=47, v1s=48 차원).

        순서는 학습의 ObservationsCfg.PolicyCfg 선언 순서와 반드시 같아야
        한다. _make_frame_spec / VARIANT_SPEC 주석 참조.
        """
        assert self.last_imu is not None
        assert self.last_gravity is not None

        imu_w = np.array([
            self.last_imu.angular_velocity.x,
            self.last_imu.angular_velocity.y,
            self.last_imu.angular_velocity.z,
        ], dtype=np.float32)
        w_b = (self.base_ang_vel_scale * imu_w).astype(np.float32)

        # 중력벡터에 가상 오프셋을 뺀다.
        # 부호 규약: +x 오프셋 -> 정책이 "앞으로 숙였다" 고 인지 -> 뒤로 보정.
        # 실기가 실제로 앞으로 쏠리는 상황이면 +x 를 주어 세운다.
        # CSV 에는 오프셋 적용 "전" 원값이 기록되므로, 로그 분석 시
        # 실제 자세를 그대로 볼 수 있다.
        g_b = np.array([
            self.last_gravity.vector.x,
            self.last_gravity.vector.y,
            self.last_gravity.vector.z,
        ], dtype=np.float32) - self.grav_offset

        gait_obs = self.gait.observation(self.gait_frequency_scale)

        q_rel = (self.q12 - self.default_q12).astype(np.float32)
        qd_rel = (self.qd12 - self.default_qd12).astype(np.float32)
        qd_rel_scaled = (self.joint_vel_scale * qd_rel).astype(np.float32)

        frame = np.concatenate(
            [w_b, g_b, vel_cmd, gait_obs, q_rel, qd_rel_scaled, self.last_action12],
            axis=0,
        ).astype(np.float32)

        assert frame.shape == (self.frame_dim,), frame.shape
        return frame

    # ──────────────────────────────────────
    # 발행
    # ──────────────────────────────────────
    def publish_joint_goals(self, q_target12: np.ndarray):
        msg = JointState()
        if self.last_joint_state is not None:
            msg.header.stamp = self.last_joint_state.header.stamp
        else:
            msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = POLICY_JOINTS.copy()
        msg.position = [float(x) for x in q_target12]
        self.goal_pub.publish(msg)

    def publish_gait_debug(self):
        """위상/주파수/상태를 디버그 토픽으로 발행.

        Foxglove 에서 phase 가 톱니파를 그리는지, 정지 명령 후 0 으로
        수렴하는지 눈으로 확인할 수 있다.
        """
        m = Float32()
        m.data = float(self.gait.phase)
        self.phase_pub.publish(m)

        m2 = Float32()
        m2.data = float(self.gait.effective_frequency)
        self.freq_pub.publish(m2)

        s = String()
        if self.gait.is_walking:
            speed = float(np.linalg.norm(self.cmd[:3]))
            s.data = "MARCHING" if speed < self.standing_threshold else "WALKING"
        elif self.gait.is_stopping:
            s.data = "STOPPING"
        else:
            s.data = "STOPPED"
        self.state_pub.publish(s)

    # ──────────────────────────────────────
    # 메인 제어 루프
    # ──────────────────────────────────────
    def control_tick(self):
        # 진단 상태(_dbg_run_state / _dbg_publishing)는 CSV 마지막 3컬럼에
        # 기록된다. 킥 전후로 gait 가 멈췄다 다시 도는지, 그 사이 cmd 가
        # 0 이었는지를 한 파일에서 보려면 '발행하지 않는 구간'도 남아야 한다.
        self._dbg_run_state = "RUN"
        self._dbg_publishing = 1
        if not self.ready():
            return
        if not self.policy_enabled():
            # robot_state 가 NORMAL 이 아니어서 막힌 구간.
            self._dbg_run_state = "GATED"
            self._dbg_publishing = 0
            self._log_yield_row()
            return

        # ── 킥 양보 게이트 ──
        # 양보 중에는 아무것도 발행하지 않는다(default 자세도 금지).
        # walkkick_v1 러너가 /joint_goals 의 유일한 주인이어야 한다.
        if self._kick_yield:
            # 워치독: /kick_done 이 오지 않는 경우(킥 러너 종료, 트리거가
            # OOD 게이트에서 거부됨 등)를 대비해 강제 복귀한다. 이게 없으면
            # 로봇이 힘 빠진 채 방치된다.
            if (self.kick_yield_timeout_s > 0.0
                    and self._kick_yield_since is not None
                    and (time.time() - self._kick_yield_since)
                    >= self.kick_yield_timeout_s):
                self.get_logger().warn(
                    f"[kick] 워치독 {self.kick_yield_timeout_s:.1f}s 초과 — "
                    "/kick_done 미수신. 강제 복귀합니다. "
                    "(킥 러너가 떠 있는지, 트리거가 거부되지 않았는지 확인)")
                self._resume_from_kick("워치독 타임아웃")
                # 복구 신호. _resume_from_kick 이 이미 _kick_yield=False 로
                # 내렸으므로, 이 메시지를 자기 자신이 되받아도 cb_kick_done
                # 의 "양보 중이 아니면 무시" 가드에 걸려 무해하다.
                self.kick_done_pub.publish(Bool(data=True))
                self.get_logger().warn(
                    "[kick] /kick_done 복구 신호 발행 — visual_servo/FSM 도 "
                    "함께 정상 상태로 되돌린다.")
            else:
                # 양보 중 — 발행은 안 하지만 로그는 남긴다. 이 구간이 비면
                # 킥 전후 gait/cmd 추이가 CSV 에서 끊겨 원인 추적이 막힌다.
                self._dbg_run_state = "YIELD"
                self._dbg_publishing = 0
                self._log_yield_row()
                return

        try:
            self.q12, self.qd12 = self.extract_q_qd_from_joint_state()
        except Exception as e:
            self.get_logger().warn(f"extract_q_qd failed: {e}")
            return

        elapsed = time.time() - self.start_wall_time

        # ── hold 판정 ──
        in_startup_hold = elapsed < self.startup_hold_seconds
        in_resume_hold = (
            self.resume_time is not None
            and self._resume_hold_dur > 0.0
            and (time.time() - self.resume_time) < self._resume_hold_dur
        )
        in_hold = in_startup_hold or in_resume_hold
        # 진단: hold 구간(정책 출력 무시, default 자세 유지)을 표시한다.
        #   킥 복귀 직후 hold 가 걸려 있으면 cmd 가 0 으로 강제되고 gait 도
        #   멈추므로, '멈췄다 풀리는' 증상의 후보가 된다.
        if in_hold:
            self._dbg_run_state = ("HOLD_START" if in_startup_hold
                                   else "HOLD_RESUME")

        # ── 속도 명령 ──
        # hold 동안 학습(generated_commands_with_reset_hold)과 동일하게 0.
        vel_cmd = np.zeros(3, dtype=np.float32) if in_hold else self.cmd.copy()

        # ── 보행 위상 갱신 ──
        # ★ 프레임을 만들기 "전" 에 갱신해야 학습과 타이밍이 맞는다 ★
        # 학습에서는 CommandManager 가 ObservationManager 보다 먼저 돌기
        # 때문이다. 순서를 바꾸면 위상이 한 스텝 뒤처진다.
        freq = self.resolve_gait_frequency(vel_cmd, in_hold)
        self.gait.update(self.policy_dt, freq)

        current_frame = self.one_frame(vel_cmd)

        if not self.history_initialized:
            self.history.reset(current_frame)
            self.history_initialized = True

        if in_hold:
            q_hold = (self.default_q12.copy()
                      if self.use_default_pose_on_startup_hold
                      else self.q12.copy())
            self.publish_joint_goals(q_hold)
            self.history.push(current_frame)
            self.publish_gait_debug()
            # hold 구간도 CSV 에 남긴다. 예전에는 여기서 그냥 return 해서
            # hold 가 걸린 사실이 로그에 안 남았고, '킥 후 1초 멈춤'의
            # 원인을 찾는 데 시간이 걸렸다. publishing=1 로 표시해 두면
            # '발행은 하는데 default 자세로 굳은' 상태가 바로 구분된다.
            self._dbg_publishing = 1
            self._log_yield_row()
            return

        frames = self.history.get_frames()
        obs = build_obs_from_frames(frames, self.frame_slices, self.obs_dim)

        obs_input = obs.reshape(1, -1).astype(np.float32)

        t0 = time.perf_counter()
        outputs = self.session.run([self.output_name], {self.input_name: obs_input})
        inference_ms = (time.perf_counter() - t0) * 1000.0

        a_raw = outputs[0].flatten().astype(np.float32)
        if a_raw.shape != (12,):
            self.get_logger().error(f"actor output dim invalid: {a_raw.shape}")
            return

        a_used = np.clip(a_raw, -self.clip_actions, self.clip_actions).astype(np.float32)

        if self.enable_action_filter:
            self.filt_action12 = (
                self.alpha * self.filt_action12 + (1.0 - self.alpha) * a_used
            ).astype(np.float32)
            a_cmd = self.filt_action12.copy()
        else:
            a_cmd = a_used.copy()

        q_target = self.default_q12 + self.action_scale * a_cmd

        # 골반 pitch 오프셋 (좌우 관절 축이 미러링되어 있어 부호가 반대)
        q_target[0] += self.pelvis_pitch_offset
        q_target[6] -= self.pelvis_pitch_offset

        q_target = np.clip(
            q_target,
            self.default_q12 - self.max_target_abs_from_default,
            self.default_q12 + self.max_target_abs_from_default,
        ).astype(np.float32)

        q_target = np.clip(
            q_target,
            self.q12 - self.max_target_delta_from_current,
            self.q12 + self.max_target_delta_from_current,
        ).astype(np.float32)

        self.publish_joint_goals(q_target)
        self.publish_gait_debug()

        # ── CSV ──
        q_rel = (self.q12 - self.default_q12).astype(np.float32)
        qd_scaled = (self.joint_vel_scale * (self.qd12 - self.default_qd12)).astype(np.float32)
        self._log_csv_row(
            vel_cmd=vel_cmd,
            q_rel=q_rel,
            qd_scaled=qd_scaled,
            a_raw=a_raw,
            a_clip=a_used,
            q_target=q_target,
            inference_ms=inference_ms,
        )

        self.last_action12 = a_used.copy()
        # 다음 프레임은 갱신된 last_action 을 반영해 다시 만든다.
        # (위상은 이미 이번 스텝에서 갱신했으므로 다시 돌리지 않는다)
        next_frame = self.one_frame(vel_cmd)
        self.history.push(next_frame)

        self.tick_count += 1

        if self.tick_count % 100 == 0:
            elapsed_total = time.time() - self.start_wall_time
            actual_hz = self.tick_count / max(1e-6, elapsed_total)
            self.get_logger().info(
                f"[rate] actual={actual_hz:.1f}Hz declared={self.policy_rate_hz}Hz "
                f"policy_dt={self.policy_dt*1000:.1f}ms")

        if self.debug_print_every > 0 and (self.tick_count % self.debug_print_every) == 0:
            state = ("WALK" if self.gait.is_walking
                     else ("STOPPING" if self.gait.is_stopping else "STOPPED"))
            self.get_logger().info(
                f"tick={self.tick_count} {state} "
                f"phase={self.gait.phase:.3f} f={self.gait.effective_frequency:.2f} "
                f"a0={a_used[0]:+.3f} qt0={q_target[0]:+.3f} "
                f"infer={inference_ms:.2f}ms")

    # ──────────────────────────────────────
    # CSV 로거
    # ──────────────────────────────────────
    def _init_csv_logger(self):
        out_dir = os.path.expanduser(self.get_parameter("csv_output_dir").value)
        os.makedirs(out_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(out_dir, f"{self.variant}_log_{timestamp}.csv")

        self._csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)

        header = ["tick", "time_sec"]
        header += ["imu_wx", "imu_wy", "imu_wz"]
        header += ["imu_ax", "imu_ay", "imu_az"]
        header += ["grav_x", "grav_y", "grav_z"]
        header += ["cmd_vx", "cmd_vy", "cmd_wz"]
        # 보행 위상 관련 (v1s 신규)
        header += ["gait_phase", "gait_cos", "gait_sin", "gait_freq", "gait_state"]

        for prefix in ["q", "qd", "q_rel", "qd_scaled", "a_prev",
                       "a_raw", "a_clip", "q_target", "q_default"]:
            for jn in POLICY_JOINTS:
                header.append(f"{prefix}_{jn}")

        # ── 킥 핸드오프 진단 컬럼 (2026-08-19 추가) ──
        #   run_state : RUN(정상) / YIELD(킥에 양보, 발행 중단)
        #               / HOLD(복귀 ease-in) / GATED(robot_state 등으로 정지)
        #   publishing: 이 틱에 /joint_goals 를 실제로 냈는지(0/1)
        #   킥 전후로 gait 가 멈췄다 다시 도는지, 그 사이 cmd 가 0 이었는지를
        #   한 파일에서 보려면 양보 구간에도 행이 남아야 한다.
        header += ["inference_ms"]
        header += ["run_state", "publishing", "robot_state"]

        self._csv_writer.writerow(header)
        self._csv_file.flush()
        self.get_logger().info(f"CSV logging: {csv_path} ({len(header)} columns)")

    def _log_yield_row(self):
        """발행하지 않는 구간(양보/게이트)에도 CSV 한 줄을 남긴다.

        정상 틱은 _log_csv_row 가 관측·액션까지 전부 남기지만, 양보 중에는
        추론 자체를 건너뛰므로 그 함수를 못 쓴다. 그렇다고 아무것도 안 남기면
        킥 전후로 gait 와 cmd 가 어떻게 변했는지가 CSV 에서 통째로 비어,
        '킥 직후 멈췄다 풀린다' 같은 증상을 추적할 수 없다.

        여기서는 시간·cmd·gait·상태만 남기고 관측/액션 컬럼은 빈칸으로 둔다.
        (헤더 길이는 맞춰야 하므로 개수만큼 '' 를 채운다)
        """
        if not self.enable_csv_log or self._csv_writer is None:
            return
        try:
            gait_obs = self.gait.observation(self.gait_frequency_scale)
            gait_state = ("WALKING" if self.gait.effective_frequency > 0.0
                          else ("STOPPING" if self.gait.is_stopping else "STOPPED"))
            row = [self.tick_count,
                   f"{time.time() - self.start_wall_time:.4f}"]
            row += ["", "", ""]                      # imu_w
            row += ["", "", ""]                      # imu_a
            row += ["", "", ""]                      # grav
            row += [f"{v:.6f}" for v in self.cmd]    # cmd (구독값 그대로)
            row += [f"{self.gait.phase:.6f}", f"{gait_obs[0]:.6f}",
                    f"{gait_obs[1]:.6f}",
                    f"{self.gait.effective_frequency:.4f}", gait_state]
            row += [""] * (9 * len(POLICY_JOINTS))   # q~q_default
            row += [""]                              # inference_ms
            row += [self._dbg_run_state, self._dbg_publishing, self.robot_state]
            self._csv_writer.writerow(row)
            self._csv_file.flush()
        except Exception:   # noqa: BLE001
            pass

    def _log_csv_row(self, vel_cmd, q_rel, qd_scaled, a_raw, a_clip, q_target, inference_ms):
        if not self.enable_csv_log or self._csv_writer is None:
            return

        elapsed = time.time() - self.start_wall_time
        row = [self.tick_count, f"{elapsed:.4f}"]

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

        if self.last_gravity is not None:
            row += [
                f"{self.last_gravity.vector.x:.6f}",
                f"{self.last_gravity.vector.y:.6f}",
                f"{self.last_gravity.vector.z:.6f}",
            ]
        else:
            row += [""] * 3

        row += [f"{vel_cmd[0]:.4f}", f"{vel_cmd[1]:.4f}", f"{vel_cmd[2]:.4f}"]

        gait_obs = self.gait.observation(self.gait_frequency_scale)
        gait_state = ("WALKING" if self.gait.is_walking
                      else ("STOPPING" if self.gait.is_stopping else "STOPPED"))
        row += [
            f"{self.gait.phase:.6f}",
            f"{gait_obs[0]:.6f}",
            f"{gait_obs[1]:.6f}",
            f"{self.gait.effective_frequency:.4f}",
            gait_state,
        ]

        for arr in [self.q12, self.qd12, q_rel, qd_scaled, self.last_action12,
                    a_raw, a_clip, q_target, self.default_q12]:
            for v in arr:
                row.append(f"{v:.6f}")

        row.append(f"{inference_ms:.3f}")

        row += [getattr(self, "_dbg_run_state", "RUN"),
                getattr(self, "_dbg_publishing", 1),
                self.robot_state]
        self._csv_writer.writerow(row)
        self._csv_row_count += 1

        if self._csv_row_count % self.csv_flush_every == 0:
            self._csv_file.flush()

    def _close_csv(self):
        if self._csv_closed or self._csv_file is None:
            return
        try:
            self._csv_file.flush()
            os.fsync(self._csv_file.fileno())
            self._csv_file.close()
            self.get_logger().info(f"CSV closed: {self._csv_row_count} rows")
        except Exception:
            pass
        self._csv_closed = True

    def destroy_node(self):
        self._close_csv()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ActorOnnxWalkingNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
