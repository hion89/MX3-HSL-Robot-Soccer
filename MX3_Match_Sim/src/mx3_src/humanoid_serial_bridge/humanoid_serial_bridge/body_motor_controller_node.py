#!/usr/bin/env python3
"""
body_motor_controller_node.py
=============================

ROS 2 노드: 전신 다이나믹셀 모터 제어 (U2D2 / Protocol 2.0).

다리/팔 18개 + 목 2개, 총 20개 모터를 SyncRead/SyncWrite 로 제어한다.

(2026-08) 목 모터 이관:
  AX-18A(Protocol 1.0, STM32 MCU 버스) → MX-64(Protocol 2.0, U2D2 버스)
  목이 U2D2 버스로 넘어오면서 이 노드가 목 모터도 함께 소유한다.
  목 전용 특수 경로(/neck/init_cmd 발행, /neck/torque_enable 발행,
  neck_state_parser_node)는 전부 제거되었고, 목은 servo_map 에 등록된
  관절 2개로서 다른 관절과 동일하게 처리된다.
  deg 단위 API(/neck/yaw_deg 등)는 neck_control_node 가 rad 로 변환해
  /joint_goals 에 실어주므로 상위 노드는 무변경이다.

이 노드는 다음만 책임진다:
  - ROS sub/pub/service 와이어링
  - /joint_goals 수신 → 캐시 → 다음 read 사이클에서 송신
  - 타이머 기반 SyncRead → /joint_states 발행
  - Behavior (standup/kick/robot_init) 서비스 핸들러
  - 진단/통계 (rate, cycle time, motor PWM/current/temp)

하드웨어 IO 는 U2D2Hardware 에 위임한다. ROS 의존성 없는 그 클래스는
unit test 시 mock 으로 바꿔치우거나 다른 도구에서 재사용 가능.

Behavior 클래스(StandUpBehavior 등)들은 `hw` 인자로 이 노드를 받는다.
behavior 가 사용하는 API surface:
  hw.has_joint, hw.id_to_joint_name
  hw.send_goals, hw.set_torque_all, hw.set_profile_velocity_all
  hw.now_sec, hw.log_info, hw.log_warn
이 7개 메서드는 이 노드에 그대로 유지된다 (대부분 U2D2Hardware 로 위임).
"""
import json
import math
import struct
import time
import threading
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, \
    QoSHistoryPolicy
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float32, String, Bool
from std_srvs.srv import Trigger

# [2026-08 CPU 최적화] projected gravity 파생 계산 헬퍼.
#   /imu/projected_gravity(200Hz) 구독을 없애고 /imu/data_50hz 의 orientation
#   에서 직접 만든다. 원본(imu_parser_node.MadgwickAHRS.gravity_vector_body)
#   과 수식·연산순서가 동일해 값이 비트 단위로 같다.
from humanoid_serial_bridge.imu_math import projected_gravity_from_quat

from humanoid_behavior_interfaces.srv import (
    RobotInit, TorqueEnable, TorqueEnableJoints, GetJointLimits,
    StandUp, MotorReset, Kick,
)
from humanoid_serial_bridge.hardware.motor_gains import GainProfile
from humanoid_serial_bridge.behaviors.robot_init_behavior import RobotInitBehavior
from humanoid_serial_bridge.behaviors.standup_behavior import StandUpBehavior
from humanoid_serial_bridge.behaviors.kick_behavior import KickBehavior

from humanoid_serial_bridge.hardware.u2d2_hardware import (
    U2D2Hardware, U2D2Config, ControlTable,
)


class BodyMotorController(Node):
    # ──────────────────────────────────────────────────────────────────
    # Motor Diagnostics (MX-64AR / Protocol 2.0)
    #   addr 124 (2B signed)  : Present PWM       (-885 ~ +885)
    #   addr 126 (2B signed)  : Present Current   (unit: 3.36 mA)
    #   addr 144 (2B unsigned): Present Input Voltage (unit: 0.1 V)
    #   addr 146 (1B unsigned): Present Temperature   (unit: 1 °C)
    # ──────────────────────────────────────────────────────────────────
    _DIAG_ADDR_START = 124
    _DIAG_READ_LEN   = 146 - 124 + 1   # 23 bytes (124..146)
    # [주의] 전류 단위는 모델마다 다르다. 목이 MX-64 로 합류하면서 한 버스에
    #        모델이 섞였다.
    #          MX-64  : 3.36 mA/LSB   ← 현재 이 상수
    #          XM540  : 2.69 mA/LSB
    #        진단 표시 전용이라 제어에는 영향이 없어 일단 그대로 둔다.
    #        전류 기반 판정을 넣게 되면 그때 관절별 단위 테이블로 분리할 것.
    _CURRENT_UNIT_MA = 3.36

    def __init__(self):
        super().__init__("body_motor_controller")

        # ──────────────────────────────────────
        # Parameters
        # ──────────────────────────────────────
        self.declare_parameter("map_file", "")
        self.declare_parameter("device_name", "/dev/ttyUSB0")
        self.declare_parameter("baudrate", 3000000)
        self.declare_parameter("read_rate_hz", 50.0)

        # Control table (Protocol 2.0 / MX-64(2.0) / X-series)
        self.declare_parameter("addr_torque_enable", 64)
        self.declare_parameter("addr_goal_position", 116)
        self.declare_parameter("addr_present_velocity", 128)
        self.declare_parameter("addr_present_position", 132)
        self.declare_parameter("addr_profile_velocity", 112)
        self.declare_parameter("addr_hardware_error", 70)
        # 토크오프 버튼 누를 때, HW 에러 래치된(=죽은) 모터를 REBOOT 로 리셋할지.
        # True: 토크오프 경로에서 에러 읽고 → 에러난 모터 reboot → 전체 토크오프.
        #       (에러 없으면 일반 토크오프와 동일하게 빠르게 처리)
        self.declare_parameter("reset_errored_on_torque_off", True)

        self.declare_parameter("len_goal_position", 4)
        self.declare_parameter("len_present_velocity", 4)
        self.declare_parameter("len_present_position", 4)
        self.declare_parameter("len_profile_velocity", 4)

        self.declare_parameter("bulk_read_start_addr", 128)
        self.declare_parameter("bulk_read_length", 8)
        self.declare_parameter("rpm_per_tick_profile", 0.229)
        # Profile Velocity 상한 (모터 Velocity Limit 과 동일하게). 이 값을 넘는 쓰기는
        # 거부되지 않고 이 값으로 clamp 된다. 보행 속도(robot_final_rpm 45.0)와 일관.
        # 모터 한계 45.2rpm → 0.2 여유로 45.0.
        self.declare_parameter("profile_velocity_limit_rpm", 45.0)

        self.declare_parameter("standup_keyframe_yaml", "")
        self.declare_parameter("kick_keyframe_yaml", "")
        self.declare_parameter("diag_rate_hz", 2.0)

        # /joint_states 의 header.stamp 를 무엇으로 찍을지.
        #   True (기본) : SyncRead 를 '시작한' 시각  ← 실측 시각에 가장 가깝다
        #   False       : SyncRead 가 '끝난' 시각    ← 구 동작(= txrx 만큼 미래로 밀림)
        # 목이 U2D2 로 합쳐지기 전 neck_state_parser_node 가 하던 back-date 를
        # 이 노드가 이어받는 셈이다. 자세한 배경은 read_and_publish() 주석 참조.
        self.declare_parameter("stamp_at_read_start", True)
        self._stamp_at_read_start = bool(
            self.get_parameter("stamp_at_read_start").value)

        # ── PID 게인 프로파일 ──
        # 게인(addr 80/82/84)은 RAM 영역이라 전원 OFF / 모터 REBOOT 시
        # EEPROM 기본값(P=800, I=0, D=0)으로 돌아간다. 그래서 기동 시와
        # 리부트 직후에 이 파일을 다시 써야 한다.
        # 빈 문자열이면 게인을 건드리지 않는다(= 기존 동작 유지).
        self.declare_parameter("gain_file", "")
        # 기입 후 읽어서 대조할지. 기동 시 1회만 하므로 비용은 무시할 수준이다.
        # 게인이 안 들어갔는데 모르고 튜닝하면 결과 해석이 전부 어긋난다.
        self.declare_parameter("gain_verify", True)

        # ──────────────────────────────────────
        # Hardware (U2D2 SDK 래퍼)
        # ──────────────────────────────────────
        ct = ControlTable(
            torque_enable        = int(self.get_parameter("addr_torque_enable").value),
            goal_position        = int(self.get_parameter("addr_goal_position").value),
            present_velocity     = int(self.get_parameter("addr_present_velocity").value),
            present_position     = int(self.get_parameter("addr_present_position").value),
            profile_velocity     = int(self.get_parameter("addr_profile_velocity").value),
            hardware_error       = int(self.get_parameter("addr_hardware_error").value),
            len_goal_position    = int(self.get_parameter("len_goal_position").value),
            len_present_velocity = int(self.get_parameter("len_present_velocity").value),
            len_present_position = int(self.get_parameter("len_present_position").value),
            len_profile_velocity = int(self.get_parameter("len_profile_velocity").value),
            bulk_read_start_addr = int(self.get_parameter("bulk_read_start_addr").value),
            bulk_read_length     = int(self.get_parameter("bulk_read_length").value),
        )
        hw_cfg = U2D2Config(
            device_name          = self.get_parameter("device_name").value,
            baudrate             = int(self.get_parameter("baudrate").value),
            map_file             = self.get_parameter("map_file").value,
            control_table        = ct,
            rpm_per_tick_profile = float(self.get_parameter("rpm_per_tick_profile").value),
            profile_velocity_limit_rpm = float(self.get_parameter("profile_velocity_limit_rpm").value),
        )
        self.hardware = U2D2Hardware(
            hw_cfg,
            log_info = self.get_logger().info,
            log_warn = self.get_logger().warn,
        )
        self.hardware.open()
        # NOTE: 자동 토크 ON 은 안전상 하지 않음 (cb_torque_enable 로 사용자가 명시 ON)

        # ──────────────────────────────────────
        # PID 게인 (RAM) — 기동 시 1회 적용
        # ──────────────────────────────────────
        # 토크를 켜기 "전"에 쓴다. 첫 토크 ON 순간부터 의도한 게인으로
        # 동작하게 하기 위함이다. (게인은 RAM 이라 토크 상태와 무관하게
        # 쓸 수 있지만, 순서를 이렇게 두면 과도 구간이 없다)
        self.gain_profile = GainProfile(
            path=str(self.get_parameter("gain_file").value),
            log_info=self.get_logger().info,
            log_warn=self.get_logger().warn,
        )
        self._apply_gains(reason="startup")

        # 튜닝용: YAML 을 고친 뒤 노드 재시작 없이 재적용한다.
        # 토크를 끄지 않으므로 로봇이 서 있는 채로, 보행 중에도 바꿀 수 있다.
        #   ros2 service call /body_motor_controller/reload_gains \
        #       std_srvs/srv/Trigger
        self.srv_reload_gains = self.create_service(
            Trigger, "~/reload_gains", self._cb_reload_gains)

        # ──────────────────────────────────────
        # Goal cache (latest-only, applied in read_and_publish slot)
        # ──────────────────────────────────────
        self._goal_lock = threading.Lock()
        self._pending_goals = None         # dict{name: rad}
        self._pending_goals_stamp = None   # rclpy.time.Time
        self.goal_positions = {}

        # ──────────────────────────────────────
        # ROS Pub / Sub
        # ──────────────────────────────────────
        self.create_subscription(JointState, "/joint_goals", self.cb_goals, 10)
        self.create_subscription(Float32, "/u2d2/set_profile_velocity",
                                 self.cb_set_profile_velocity, 10)
        # 관절별 개별 Profile Velocity.
        #   위 /u2d2/set_profile_velocity 가 "전체 같은 속도" 라면 이건 "선택 개별 속도".
        #   목처럼 몸통과 전혀 다른 속도(≈10rpm)로 움직여야 하는 관절 때문에 추가.
        #   메시지 규약: JointState 를 재활용한다 (새 인터페이스 패키지 재빌드 회피)
        #     name[]     : 대상 관절 이름
        #     velocity[] : 해당 관절의 Profile Velocity [rpm]
        #   예) ros2 topic pub --once /u2d2/set_joint_profile_velocity \
        #         sensor_msgs/JointState \
        #         "{name: ['neck_yaw_joint'], velocity: [10.0]}"
        self.create_subscription(JointState, "/u2d2/set_joint_profile_velocity",
                                 self.cb_set_joint_profile_velocity, 10)
        # 관절별 Goal PWM(100) 상한 런타임 설정.
        #   motor_gains.yaml 의 goal_pwm 은 '평상시 운용값' 이고, behavior
        #   (robot_init/standup)는 진입 시 자동으로 100% 로 올린다. 그런데
        #   키프레임 편집 GUI 처럼 behavior 를 안 거치고 /joint_goals 로 직접
        #   자세를 잡는 도구는 그 승격 경로를 못 탄다. 팔이 운용값(예: 100,
        #   =11.3%)에 묶여 목표까지 이동하지 못한다(실측 이동 하한 150).
        #   → 그런 도구가 스스로 상한을 올릴 수 있는 경로를 연다.
        #
        #   메시지 규약: JointState 재활용 (새 인터페이스 패키지 재빌드 회피).
        #     name[]   : 대상 관절 이름
        #     effort[] : 해당 관절의 Goal PWM. 0 이하면 그 모터의 PWM Limit(=100%).
        #   예) 팔 전체를 100% 로:
        #     ros2 topic pub --once /u2d2/set_joint_goal_pwm sensor_msgs/JointState \
        #       "{name: ['l_shoulder_pitch_joint','r_shoulder_pitch_joint'], \
        #         effort: [0.0, 0.0]}"
        #
        #   ★ 되돌리려면 reload_gains 를 부르면 된다(YAML 운용값 재적용).
        #     ros2 service call /body_motor_controller/reload_gains \
        #         std_srvs/srv/Trigger
        self.create_subscription(JointState, "/u2d2/set_joint_goal_pwm",
                                 self.cb_set_joint_goal_pwm, 10)
        self.js_pub = self.create_publisher(JointState, "/joint_states", 10)

        # Main read timer
        read_rate_hz = float(self.get_parameter("read_rate_hz").value)
        self.read_period_sec = 1.0 / max(1e-6, read_rate_hz)
        self.create_timer(self.read_period_sec, self.read_and_publish)

        # Rate print timer
        self.timestamps = []
        self.last_js = None
        self.create_timer(1.0, self.print_rate)

        # ──────────────────────────────────────
        # Behaviors (self 를 hw 로 주입)
        # ──────────────────────────────────────

        # ── 목(Neck) 관련 특수 경로는 전부 제거됨 (2026-08) ──
        # 목 모터가 AX-18A(Protocol 1.0, STM32 MCU 버스) → MX-64(Protocol 2.0,
        # U2D2 버스)로 이관되면서, 목도 그냥 servo_map 에 등록된 관절 2개일
        # 뿐이다. 즉:
        #   - 목표 위치 : /joint_goals 로 들어와 send_goals() 가 처리
        #   - 현재 위치 : SyncRead 로 같이 읽혀 /joint_states 로 나감
        #                 (기존 neck_state_parser_node 는 삭제됨)
        #   - 토크      : set_torque_all() 이 ID 19/20 까지 자동으로 덮음
        # 따라서 예전의 /neck/init_cmd, /neck/torque_enable 퍼블리셔와
        # _send_neck_init() 헬퍼는 전부 삭제했다.
        # deg 단위 API(/neck/yaw_deg 등)는 neck_control_node 가 rad 로 바꿔
        # /joint_goals 에 실어주므로, 상위 노드(전략/시선추적)는 무변경이다.
        torque_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub_torque_enabled = self.create_publisher(
            Bool, '/u2d2/torque_enabled', torque_qos)
        self._torque_enabled = False
        self._publish_torque_enabled(False, 'startup')

        self.robot_init = RobotInitBehavior(hw=self)
        self.srv_robot_init = self.create_service(
            RobotInit, "/u2d2/robot_init", self.cb_robot_init)
        self.srv_motor_zero = self.create_service(
            RobotInit, "/u2d2/motor_zero", self.cb_motor_zero)
        self.srv_torque = self.create_service(
            TorqueEnable, "/u2d2/torque_enable", self.cb_torque_enable)
        # 특정 조인트만 토크 ON/OFF (kinesthetic teaching 등에 사용).
        # /u2d2/torque_enable 이 "전체" 라면 이건 "선택" 버전.
        self.srv_torque_joints = self.create_service(
            TorqueEnableJoints, "/u2d2/torque_enable_joints",
            self.cb_torque_enable_joints)
        # 모든 조인트의 EEPROM Min/Max Position Limit 를 rad 단위로 조회.
        # pose_slider_gui 가 시작 시 호출해서 슬라이더 범위 자동 설정.
        self.srv_get_limits = self.create_service(
            GetJointLimits, "/u2d2/get_joint_limits",
            self.cb_get_joint_limits)

        # standup (중력 z 추적 필요)
        self.last_gravity_z = -1.0
        # ── [2026-08 CPU 최적화] /imu/projected_gravity 200Hz → /imu/data_50hz ──
        #   _cb_grav_standup 은 float 하나(last_gravity_z)를 저장할 뿐이고,
        #   그 값은 _cb_standup(StandUp 서비스 핸들러) 안에서만 읽힌다.
        #   즉 '낙상 후 일어설 때 한 번' 쓰는 값을 위해 초당 200번 깨어나고 있었다.
        #   rclpy 는 깨어날 때마다 wait set 을 재구축하므로 그게 그대로 CPU 다.
        #
        #   중력벡터는 /imu/data.orientation 만의 순수 함수라 별도 토픽이 필요없다
        #   (imu_math.projected_gravity_from_quat — 원본과 비트 단위로 동일).
        #   → 50Hz IMU 하나만 받아 파생 계산한다. 깨우기 200 → 50.
        #
        #   지연 영향: standup 판정은 '엎드렸나 누웠나'를 보는 것이고 낙상은
        #   ~300ms 에 걸쳐 일어난다. 15ms 지연은 무의미하다.
        #   되돌리려면 launch 에서 gravity_imu_topic:=/imu/data 로 지정.
        self.declare_parameter("gravity_imu_topic", "/imu/data_50hz")
        self._grav_imu_topic = str(self.get_parameter("gravity_imu_topic").value)
        self.create_subscription(Imu, self._grav_imu_topic,
                                 self._cb_grav_standup, 10)

        # ── [2026-08] U2D2 반이중 경합 제거 (TX gap) ──────────────────────
        #   SyncWrite 직후 바이트가 선로로 빠져나갈 시간만큼 대기한다.
        #   벤치 실측(u2d2_txrx_bench.py, 20모터 @3Mbps):
        #       gap 0    → txrx 9.45ms
        #       gap 200us → txrx 3.03ms
        #       gap 400us → txrx 2.86ms   ← 기본값
        #   SyncWrite 114 byte @3Mbps = 0.38ms 이므로 400us 가 적정.
        #
        #   ★ 초판은 tcdrain 을 썼다가 실패했다. txrx 는 줄었지만 write 가
        #     8.5ms 로 폭증해 CycleTime 이 12.0 → 12.9ms 로 악화됐다.
        #     자세한 경위는 u2d2_hardware.tx_gap() 주석 참조.
        #
        #   0.0 으로 두면 비활성(기존 동작). 롤백은 tx_gap_us:=0.0
        self.declare_parameter("tx_gap_us", 400.0)
        self._tx_gap_us = float(self.get_parameter("tx_gap_us").value)
        self.get_logger().info(
            f"TX gap (SyncWrite→SyncRead 경합 제거): "
            f"{self._tx_gap_us:.0f} us"
            f"{' (비활성)' if self._tx_gap_us <= 0.0 else ''}")
        self.standup = StandUpBehavior(
            hw=self,
            keyframe_yaml=self.get_parameter("standup_keyframe_yaml").value)
        self.srv_standup = self.create_service(StandUp, "/u2d2/standup", self._cb_standup)

        # kick
        self.kick = KickBehavior(
            hw=self,
            keyframe_yaml=self.get_parameter("kick_keyframe_yaml").value)
        self.srv_kick = self.create_service(Kick, "/u2d2/kick", self._cb_kick)

        # ──────────────────────────────────────
        # Behavior 실행 상태 발행 (/u2d2/behavior_status)
        # ──────────────────────────────────────
        # standup/kick 같은 키프레임 동작은 서비스 호출 직후 바로 리턴하고,
        # 실제 재생은 이 노드의 제어 루프(behavior.tick)에서 비동기로 진행된다.
        # 그래서 외부(fall_manager 등)는 "재생이 언제 끝났는지"를 알 수 없다.
        # 이 토픽이 현재 실행 중인 behavior 의 상태를 밖으로 노출해,
        # fall_manager 가 'standup active True→False' 엣지를 보고 복구 판정을
        # 이벤트 기반으로 내릴 수 있게 한다. (시간 추정 X, 실제 이벤트 O)
        #
        # 타입은 std_msgs/String + JSON 페이로드. 새 커스텀 메시지(인터페이스
        # 패키지 재빌드)를 피하면서, ros2 topic echo 로 사람이 바로 읽을 수
        # 있다. 나중에 타입 메시지가 필요하면 이 발행부 한 곳만 바꾸면 된다.
        self.behavior_status_pub = self.create_publisher(
            String, "/u2d2/behavior_status", 10)
        # 20Hz 로 '현재 스냅샷'을 항상 발행한다. 늦게 구독한 노드도 곧바로
        # 최신 상태를 받으므로, 1회성 엣지를 놓쳐서 멈추는 일이 없다.
        # (제어 루프와 같은 단일스레드 executor 라 is_active 읽기에 레이스 없음)
        self.create_timer(0.05, self._publish_behavior_status)

        # motor reset
        self.srv_motor_reset = self.create_service(
            MotorReset, "/u2d2/motor_reset", self._cb_motor_reset)

        # ──────────────────────────────────────
        # Cycle time stats (last 200 cycles)
        # ──────────────────────────────────────
        self.t_cycle_ns_hist      = deque(maxlen=200)
        self.t_txrx_ns_hist       = deque(maxlen=200)
        self.t_parse_ns_hist      = deque(maxlen=200)
        self.t_pub_ns_hist        = deque(maxlen=200)
        self.t_write_ns_hist      = deque(maxlen=200)
        self.t_goal_delay_ms_hist = deque(maxlen=200)
        self.t_overrun_cnt = 0

        # ──────────────────────────────────────
        # Motor Diagnostics (PWM / Current / Voltage / Temp)
        # ──────────────────────────────────────
        self._init_diagnostics()

    # ════════════════════════════════════════════════════════════════
    # /joint_goals → 캐시
    # ════════════════════════════════════════════════════════════════
    def cb_goals(self, msg: JointState):
        """/joint_goals 수신 → 다음 read 사이클에 쓸 목표를 캐시한다.

        ★ 반드시 '관절 단위 병합' 이어야 한다 ★
          /joint_goals 는 퍼블리셔가 하나가 아니다.
            - 정책 러너(actor_onnx_*)  : 다리/팔 18관절, 50Hz
            - neck_control_node        : 목 2관절, 시선 추적 rate
            - fall_manager             : 보호 자세, 이벤트성
          예전처럼 dict 를 통째로 교체하면, 같은 read 사이클 안에서 나중에
          도착한 목 명령(2관절)이 정책이 보낸 다리 명령(18관절)을 통째로
          지워버린다. 증상은 "보행 중 가끔 한 프레임씩 다리가 안 따라옴"
          이라 원인 찾기가 대단히 어렵다.
          → 관절별 최신값을 누적해서, 한 사이클에 모든 퍼블리셔의 최신
            명령이 함께 나가도록 한다. 같은 값을 다시 써도 무해하다.
        """
        goals = {
            n: p for n, p in zip(msg.name, msg.position)
            if self.hardware.has_joint(n) and p is not None and not math.isnan(p)
        }
        if not goals:
            return
        with self._goal_lock:
            merged = dict(self._pending_goals) if self._pending_goals else {}
            merged.update(goals)
            self._pending_goals = merged
            self.goal_positions = merged
            # stamp 는 '가장 최근에 도착한 명령' 기준으로 갱신한다.
            # (GoalDelay 통계 = 마지막 명령이 실제 버스로 나가기까지의 지연)
            self._pending_goals_stamp = self.get_clock().now()

    # ════════════════════════════════════════════════════════════════
    # 메인 사이클: pending goal write → SyncRead → publish
    # ════════════════════════════════════════════════════════════════
    def read_and_publish(self):
        t0 = time.perf_counter_ns()

        # 1) pending goal write
        tw0 = time.perf_counter_ns()
        pending = None
        goal_stamp = None
        with self._goal_lock:
            if self._pending_goals is not None:
                pending = self._pending_goals
                goal_stamp = self._pending_goals_stamp
                self._pending_goals = None

        if pending is not None:
            self.hardware.send_goals(pending)
            if goal_stamp is not None:
                delay_ms = (self.get_clock().now().nanoseconds - goal_stamp.nanoseconds) * 1e-6
                self.t_goal_delay_ms_hist.append(delay_ms)

            # ── [2026-08] SyncWrite → SyncRead 반이중 경합 제거 ──────────
            #   send_goals() 는 FTDI 로 바이트를 넘기고 즉시 반환한다.
            #   그 바이트가 아직 선로에 있는 채로 아래 read_state() 가
            #   SyncRead 를 밀어넣으면 txrx 가 2.94ms → 9.45ms 로 늘어난다.
            #   (실측 근거는 u2d2_hardware.tx_gap() 주석 참조)
            #
            #   ★ 액션 지연은 늘지 않는다 — goal 은 이미 나가고 있다.
            #     실측 GoalDelay p50 8.2 → 6.6ms (오히려 개선)
            #   ★ 이 대기는 아래 tw1 계측에 포함되므로 진단 로그의
            #     'write=' 값에 gap 만큼(약 0.4ms)이 더해져 보인다. 정상이다.
            #
            #   되돌리려면 launch 에서 tx_gap_us:=0.0
            self.hardware.tx_gap(self._tx_gap_us)
        tw1 = time.perf_counter_ns()

        # 2) read state
        #
        # [타임스탬프 정합성 — 반드시 read 직전 시각을 잡아둘 것]
        #   SyncRead 는 "명령 패킷 1회 브로드캐스트 → 각 모터가 그 시점의
        #   Present Position 을 래치 → ID 순서대로 응답 패킷 직렬 전송" 이다.
        #   즉 실제 측정(래치) 시각은 txRxPacket() '시작' 시점에 가깝고,
        #   함수가 반환하는 시점은 20개 응답을 다 받아낸 뒤라서 그만큼 늦다.
        #
        #   예전 구조(목 = MCU 시리얼, neck_state_parser_node)에서 목 각도를
        #   실측 시각으로 back-date 해서 발행했던 것과 같은 이유다. 목이
        #   U2D2 로 합쳐지면서 그 back-date 로직이 함께 사라졌고, 지금은
        #   "읽기 끝난 시각" 으로 찍히고 있어 TF 가 실제보다 미래로 밀린다.
        #   목이 팬(pan) 중일 때 카메라 자세가 앞서 나가 공 위치가 옆으로
        #   흔들리는 형태로 나타난다.
        #
        #   오차 크기 = SyncRead 왕복시간(t_txrx). 1Hz 진단 로그의
        #   "CycleTime ms ... txrx=X.XXX avg" 로 실측 확인할 것.
        t_read_start = self.get_clock().now()
        t1 = time.perf_counter_ns()
        ok, readings = self.hardware.read_state()
        t2 = time.perf_counter_ns()

        if not ok:
            self.t_cycle_ns_hist.append(t2 - t0)
            self.t_txrx_ns_hist.append(t2 - t1)
            self.t_write_ns_hist.append(tw1 - tw0)
            return

        # 3) build JointState
        js = JointState()
        # ★ now() 가 아니라 read 직전 시각을 쓴다 (위 2) 주석의 back-date).
        #   robot_state_publisher 는 이 stamp 를 그대로 TF 에 실으므로,
        #   여기서 한 번 틀리면 base_link→camera 체인 전체가 같이 틀어진다.
        #   stamp_at_read_start=False 로 두면 예전 동작(읽기 완료 시각)으로
        #   되돌아간다 — A/B 비교용 탈출구.
        if self._stamp_at_read_start:
            js.header.stamp = t_read_start.to_msg()
        else:
            js.header.stamp = self.get_clock().now().to_msg()
        for r in readings:
            js.name.append(r.name)
            js.position.append(r.rad)
            js.velocity.append(r.radps)
        t3 = time.perf_counter_ns()

        # 4) publish
        self.js_pub.publish(js)
        t4 = time.perf_counter_ns()

        # 5) stats
        self.t_cycle_ns_hist.append(t4 - t0)
        self.t_write_ns_hist.append(tw1 - tw0)
        self.t_txrx_ns_hist.append(t2 - t1)
        self.t_parse_ns_hist.append(t3 - t2)
        self.t_pub_ns_hist.append(t4 - t3)
        if (t4 - t0) * 1e-9 > self.read_period_sec:
            self.t_overrun_cnt += 1

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        self.timestamps.append(now_sec)
        t_cut = now_sec - 1.0
        self.timestamps = [t for t in self.timestamps if t >= t_cut]
        self.last_js = js

        # 6) behavior tick (active 일 때만)
        if self.robot_init.is_active():
            self.robot_init.tick(self.last_js)
        if self.standup.is_active():
            self.standup.tick(self.last_js)
        if self.kick.is_active():
            self.kick.tick(self.last_js)

    # ════════════════════════════════════════════════════════════════
    # Behavior status 발행 (20Hz)
    # ════════════════════════════════════════════════════════════════
    def _publish_behavior_status(self):
        """
        현재 실행 중인 blocking behavior(robot_init/standup/kick)의 상태를
        /u2d2/behavior_status 토픽으로 JSON 발행한다.

        우선순위: robot_init > standup > kick > none.
          (서비스 핸들러가 동시 실행을 막지만, 혹시 모를 경우를 위해 우선순위)

        페이로드 예:
          {"behavior":"standup","active":true,"phase":"pushup",
           "phase_idx":1,"total_phases":5}
        소비자(fall_manager)는 behavior=="standup" && active 의 True→False
        엣지로 standup 재생 종료를 판정한다.
        """
        name = "none"
        active = False
        phase = ""
        phase_idx = -1
        total = -1

        if self.robot_init.is_active():
            name, active = "robot_init", True
            # robot_init 은 phase getter 가 없으므로 phase 정보는 비워둔다.
        elif self.standup.is_active():
            name, active = "standup", True
            phase = self.standup.get_phase_name()
            phase_idx = self.standup.get_phase_idx()
            total = self.standup.get_total_phases()
        elif self.kick.is_active():
            name, active = "kick", True
            phase = self.kick.get_phase_name()
            phase_idx = self.kick.get_phase_idx()
            total = self.kick.get_total_phases()

        payload = {
            "behavior": name,        # 실행 중 behavior 이름 (없으면 "none")
            "active": active,        # blocking 동작 진행 여부
            "phase": phase,          # 현재 phase 이름 (있을 때만)
            "phase_idx": phase_idx,
            "total_phases": total,
        }
        msg = String()
        msg.data = json.dumps(payload)
        self.behavior_status_pub.publish(msg)

    # ════════════════════════════════════════════════════════════════
    # Rate / cycle time print (1 Hz)
    # ════════════════════════════════════════════════════════════════
    def print_rate(self):
        if len(self.timestamps) > 1:
            dt = self.timestamps[-1] - self.timestamps[0]
            hz = (len(self.timestamps) - 1) / dt if dt > 0 else 0.0
        else:
            hz = 0.0
        self.get_logger().info(f"JointState rate: {hz:.1f} Hz")

        # SyncRead 성공률 (per-id)
        rt = self.hardware.read_total
        rf = self.hardware.read_fail
        fail_pct = (100.0 * rf / rt) if rt > 0 else 0.0
        self.get_logger().info(f"SyncRead fail: {rf}/{rt} ({fail_pct:.2f}%)")

        # Cycle time 통계
        if len(self.t_cycle_ns_hist) >= 5:
            cyc_ms   = np.asarray(self.t_cycle_ns_hist, dtype=np.float64) * 1e-6
            txrx_ms  = np.asarray(self.t_txrx_ns_hist, dtype=np.float64) * 1e-6
            parse_ms = np.asarray(self.t_parse_ns_hist, dtype=np.float64) * 1e-6
            pub_ms   = np.asarray(self.t_pub_ns_hist, dtype=np.float64) * 1e-6
            write_ms = (np.asarray(self.t_write_ns_hist, dtype=np.float64) * 1e-6
                        if self.t_write_ns_hist else np.zeros(1))

            def p(a, q): return float(np.percentile(a, q))
            self.get_logger().info(
                "CycleTime ms (last %d): avg=%.3f p50=%.3f p95=%.3f max=%.3f | "
                "write=%.3f avg | txrx=%.3f avg | parse=%.3f avg | pub=%.3f avg | overrun=%d"
                % (len(cyc_ms),
                   float(np.mean(cyc_ms)), p(cyc_ms, 50), p(cyc_ms, 95), float(np.max(cyc_ms)),
                   float(np.mean(write_ms)),
                   float(np.mean(txrx_ms)),
                   float(np.mean(parse_ms)),
                   float(np.mean(pub_ms)),
                   self.t_overrun_cnt)
            )

            if len(self.t_goal_delay_ms_hist) >= 3:
                gd = np.asarray(self.t_goal_delay_ms_hist, dtype=np.float64)
                self.get_logger().info(
                    "GoalDelay ms (last %d): avg=%.1f p50=%.1f p95=%.1f max=%.1f"
                    % (len(gd), float(np.mean(gd)), p(gd, 50), p(gd, 95), float(np.max(gd)))
                )

    # ════════════════════════════════════════════════════════════════
    # Motor Diagnostics
    # ════════════════════════════════════════════════════════════════
    def _init_diagnostics(self):
        diag_hz = float(self.get_parameter("diag_rate_hz").value)
        self._diag_group_read = self.hardware.make_bulk_reader(
            self._DIAG_ADDR_START, self._DIAG_READ_LEN)
        self._diag_pub = self.create_publisher(String, "/motor_diagnostics", 10)
        self.create_timer(1.0 / max(0.1, diag_hz), self._diag_tick)
        self.get_logger().info(
            f"Motor diagnostics: {diag_hz:.1f} Hz, "
            f"addr {self._DIAG_ADDR_START}–{self._DIAG_ADDR_START + self._DIAG_READ_LEN - 1} "
            f"({self._DIAG_READ_LEN}B × {self.hardware.num_motors} motors)"
        )

    def _diag_tick(self):
        if self._diag_group_read.txRxPacket() != 0:
            return

        # HW Error Status(addr 70)도 함께 읽어 진단에 실는다.
        #   - fall_manager 등이 이 토픽만 구독해도 "죽은 모터"를 알 수 있게 하기 위함.
        #   - 2Hz 에서 18모터 개별 read → 3Mbps 버스에선 부담 미미.
        hw_errors = self.hardware.read_hardware_errors()   # {id: err_byte}

        motors = []
        for dxl_id in self.hardware.used_ids:
            if not self._diag_group_read.isAvailable(
                    dxl_id, self._DIAG_ADDR_START, self._DIAG_READ_LEN):
                continue

            pwm_raw  = self._diag_group_read.getData(dxl_id, 124, 2)
            pwm      = struct.unpack('<h', struct.pack('<H', pwm_raw & 0xFFFF))[0]

            cur_raw  = self._diag_group_read.getData(dxl_id, 126, 2)
            cur_s    = struct.unpack('<h', struct.pack('<H', cur_raw & 0xFFFF))[0]
            current_ma = cur_s * self._CURRENT_UNIT_MA

            volt_raw = self._diag_group_read.getData(dxl_id, 144, 2)
            voltage_v = (volt_raw & 0xFFFF) * 0.1

            temp = self._diag_group_read.getData(dxl_id, 146, 1) & 0xFF

            motors.append({
                "id": dxl_id,
                "name": self.hardware.id_to_joint_name(dxl_id) or f"dxl_{dxl_id}",
                "pwm": pwm,
                "current_ma": round(current_ma, 1),
                "voltage_v": round(voltage_v, 1),
                "temp_c": temp,
                # addr 70 Hardware Error Status (0=정상, 비트: 과열/과부하 등).
                "hw_error": int(hw_errors.get(dxl_id, 0)),
            })

        msg = String()
        msg.data = json.dumps({"stamp": time.time(), "motors": motors})
        self._diag_pub.publish(msg)

    # ════════════════════════════════════════════════════════════════
    # Behavior service callbacks (행위는 behavior 클래스에 위임)
    # ════════════════════════════════════════════════════════════════
    def cb_set_profile_velocity(self, msg: Float32):
        self.hardware.set_profile_velocity_all(float(msg.data))

    def cb_set_joint_profile_velocity(self, msg: JointState):
        """관절별 Profile Velocity 설정 (JointState.name[] + velocity[] rpm).

        목처럼 몸통과 전혀 다른 속도로 움직여야 하는 관절을 위한 경로.
        일괄 쓰기(set_profile_velocity_all)는 servo_map 의
        profile_velocity_exclude 에 걸린 관절을 건너뛰므로, 그런 관절의
        속도는 반드시 이 토픽으로 지정해야 한다.

        길이가 안 맞거나 velocity 가 비면 무시한다 (JointState 는 velocity 를
        생략할 수 있는 메시지라 방어가 필요).
        """
        if not msg.name or len(msg.velocity) < len(msg.name):
            self.get_logger().warn(
                "/u2d2/set_joint_profile_velocity: name[] 과 velocity[] 길이가 맞지 않음 → 무시")
            return

        rpm_by_joint = {}
        for name, rpm in zip(msg.name, msg.velocity):
            if rpm is None or math.isnan(rpm) or rpm < 0.0:
                continue
            rpm_by_joint[name] = float(rpm)

        if not rpm_by_joint:
            return

        applied = self.hardware.set_profile_velocity_joints(rpm_by_joint)
        if applied:
            detail = ", ".join(f"{n}={rpm_by_joint[n]:.1f}rpm" for n in applied)
            self.get_logger().info(f"Profile Velocity (개별): {detail}")

    def cb_set_joint_goal_pwm(self, msg: JointState):
        """관절별 Goal PWM 상한 설정 (JointState.name[] + effort[]).

        effort 값이 0 이하이면 그 모터의 PWM Limit(=100%)을 뜻한다.
        GUI 가 모터별 PWM Limit 을 몰라도 "최대로" 를 표현할 수 있게 하기 위함.

        길이가 안 맞거나 effort 가 비면 무시한다 (JointState 는 effort 를
        생략할 수 있는 메시지라 방어가 필요).
        """
        if not msg.name or len(msg.effort) < len(msg.name):
            self.get_logger().warn(
                "/u2d2/set_joint_goal_pwm: name[] 과 effort[] 길이가 맞지 않음 → 무시")
            return

        pwm_by_id = {}
        unknown = []
        for name, val in zip(msg.name, msg.effort):
            if not self.hardware.has_joint(name):
                unknown.append(name)
                continue
            if val is None or math.isnan(val):
                continue
            dxl_id = int(self.hardware.jid[name])
            if val <= 0.0:
                # 0 이하 = "그 모터의 최대치". PWM Limit 은 모터마다 다를 수
                # 있으므로 캐시에서 각자의 값을 꺼낸다.
                pwm_by_id[dxl_id] = int(self.hardware.pwm_limit.get(dxl_id, 885))
            else:
                pwm_by_id[dxl_id] = int(val)

        if unknown:
            self.get_logger().warn(
                f"/u2d2/set_joint_goal_pwm: 모르는 관절 무시 {unknown}")
        if not pwm_by_id:
            return

        res = self.hardware.set_goal_pwm(pwm_by_id)
        if res["written"]:
            detail = ", ".join(
                f"{self.hardware.id_to_joint_name(i)}={v}"
                for i, v in sorted(pwm_by_id.items()))
            self.get_logger().info(f"Goal PWM (개별): {detail}")

    def cb_robot_init(self, req, res):
        if self.standup.is_active() or self.kick.is_active():
            res.success = False
            res.message = "standup/kick running"
            return res
        ok, msg = self.robot_init.start(
            init_rpm=req.init_rpm,
            final_rpm=req.final_rpm,
            tol=req.pos_tolerance_rad,
            settle_count=req.settle_count,
            timeout_sec=req.timeout_sec,
        )
        # (2026-08) 목 별도 발행 제거.
        # 목이 U2D2 버스로 넘어오면서 RobotInitBehavior 의 targets 에 목 관절이
        # 직접 포함된다 → 몸통과 같은 send_goals / 도착판정 경로를 탄다.
        # 목 속도만 몸통과 달라야 해서, behavior 가 start() 안에서
        # set_profile_velocity_joints() 로 따로 지정한다.
        res.success = bool(ok)
        res.message = str(msg)
        return res

    def cb_motor_zero(self, req, res):
        """
        모든 조인트를 0 rad 위치로 보낸다. 정비/캘리브레이션/안전 자세 복귀용.
        주의: 현재 자세에 따라 자기 간섭이 발생할 수 있으므로 로봇이 공중에
              매달려 있거나 지면에 안전하게 놓인 상태에서만 사용.
        """
        if self.standup.is_active() or self.kick.is_active():
            res.success = False
            res.message = "standup/kick running"
            return res
        ok, msg = self.robot_init.start_zero(
            init_rpm=req.init_rpm,
            final_rpm=req.final_rpm,
            tol=req.pos_tolerance_rad,
            settle_count=req.settle_count,
            timeout_sec=req.timeout_sec,
        )
        # (2026-08) 목 별도 발행 제거 — start_zero() 의 targets 에 목이 0 rad 로
        # 포함되므로 "전 조인트 0 rad" 의미가 목까지 자동으로 확장된다.
        res.success = bool(ok)
        res.message = str(msg)
        return res

    def _publish_torque_enabled(self, enabled: bool, reason: str = ""):
        self._torque_enabled = bool(enabled)
        msg = Bool()
        msg.data = self._torque_enabled
        self.pub_torque_enabled.publish(msg)
        suffix = f" ({reason})" if reason else ""
        self.get_logger().info(
            f"body torque state -> {'ON' if self._torque_enabled else 'OFF'}{suffix}")

    def cb_torque_enable(self, req, res):
        """
        전신(U2D2, ID 1~20) 토크 ON/OFF.

        (2026-08) 목이 U2D2 버스로 넘어오면서 목 전용 처리가 사라졌다.
        set_torque_all() 이 servo_map 에 등록된 모든 ID(=목 19/20 포함)를
        돌기 때문에, 예전의 /neck/torque_enable 발행이 필요 없다.
        목만 따로 끄고 싶으면 /u2d2/torque_enable_joints 를 쓴다.
        """
        try:
            # 전신 (U2D2, ID 1~20 = 다리/팔 18 + 목 2)
            if req.enable:
                # ── 토크 ON ──
                # 직전 토크 OFF 경로에서 set_torque_off_with_reset() 가
                # 죽은 모터를 REBOOT 했을 수 있다. REBOOT 는 RAM 을
                # 초기화하므로 PID 게인이 기본값으로 돌아가 있다.
                # 토크를 켜기 "전"에 다시 써서, 켜지는 순간부터 의도한
                # 게인으로 동작하게 한다.
                self._apply_gains(reason="torque_on")
                self.set_torque_all(True)
                res.success = True
                res.message = "torque EN"
            else:
                # ── 토크 OFF: 죽은(에러 래치) 모터 리셋 옵션 ──
                # 토크오프 버튼은 로봇을 내리는 의도적 동작이라, 이때 오버로드/
                # 과열로 죽은 모터를 REBOOT 로 함께 리셋한다(래치는 reboot 로만
                # 지워짐). 토크가 꺼지는 상황이라 급발진 위험이 없어 안전한 타이밍.
                if bool(self.get_parameter(
                        "reset_errored_on_torque_off").value):
                    rep = self.hardware.set_torque_off_with_reset()
                    self._publish_torque_enabled(False, 'torque service')
                    res.success = True
                    res.message = (
                        f"torque DIS | errored={list(rep['errored'])} "
                        f"rebooted={rep['rebooted']} residual={rep['residual']}")
                    if rep["residual"]:
                        self.get_logger().warn(
                            f"reboot 후에도 에러 잔존: {rep['residual']} "
                            f"(과열/저전압 등 원인 지속 — 물리 조치 필요)")
                else:
                    # 리셋 비활성 → 기존 동작(단순 전체 토크오프)
                    self.set_torque_all(False)
                    res.success = True
                    res.message = "torque DIS"
        except Exception as e:
            res.success = False
            res.message = f"torque service failed: {e}"
            return res

        return res

    def cb_torque_enable_joints(self, req, res):
        """
        특정 조인트들만 토크 ON/OFF.

        servo_map 에 등록된 모든 관절이 대상이다.
        (2026-08) 목도 U2D2 버스로 넘어왔으므로 neck_yaw_joint /
        neck_pitch_joint 도 이 서비스로 제어된다. neck_control_node 의
        /neck/torque_enable 구독이 내부적으로 이 서비스를 호출한다.

        용도:
          - pose_slider_gui 의 체크박스 → 손으로 잡고 싶은 관절만 OFF
          - kinesthetic teaching: 일부만 풀어두고 그 관절만 조정
          - 목 부호/영점 검증: 목만 토크 오프하고 손으로 돌려보기
          - 디버깅: 특정 모터만 토크 끄고 외력 영향 측정

        요청:
          joint_names : 대상 조인트 이름 목록. 빈 리스트면 success=false.
          enable      : True=토크 ON, False=토크 OFF
        응답:
          success         : 하나라도 적용됐으면 True
          message         : 처리 요약 또는 에러 메시지
          applied_joints  : 실제로 토크가 변경된 조인트 이름들 (알 수 없는
                            이름은 제외됨). GUI 가 어떤 게 처리됐는지 확인 가능.
        """
        # 입력 검증
        joint_names = list(req.joint_names) if req.joint_names else []
        if not joint_names:
            res.success = False
            res.message = "joint_names 가 비어 있음"
            res.applied_joints = []
            return res

        # 진행 중인 behavior 와 충돌 방지.
        # behavior 가 토크를 켜둔 상태로 동작하는데 도중에 일부만 끄면
        # 자세가 무너지거나 다음 명령에서 이상 동작이 날 수 있다.
        if (self.robot_init.is_active()
                or self.standup.is_active()
                or self.kick.is_active()):
            res.success = False
            res.message = "behavior 가 실행 중 (robot_init/standup/kick) — 끝난 후 다시 시도"
            res.applied_joints = []
            return res

        applied = []
        skipped = []
        failed = []
        for jname in joint_names:
            # joint_name 검증. 목 조인트나 오타가 들어오면 skipped.
            if not self.hardware.has_joint(jname):
                skipped.append(jname)
                continue
            # joint_name → dxl_id
            dxl_id = self.hardware.jid[jname]
            try:
                # hardware.torque_enable 는 단일 모터 write1byte.
                # 이미 같은 상태여도 멱등 — write 비용은 미미.
                self.hardware.torque_enable(dxl_id, bool(req.enable))
                applied.append(jname)
            except Exception as e:
                failed.append(f"{jname}({e})")

        # 결과 종합
        state_str = "EN" if req.enable else "DIS"
        if not applied:
            res.success = False
            res.message = (
                f"적용된 조인트 없음. skipped={skipped}, failed={failed}"
            )
        else:
            res.success = True
            res.message = (
                f"torque {state_str}: {len(applied)} joints"
                + (f", skipped={skipped}" if skipped else "")
                + (f", failed={failed}" if failed else "")
            )
            self.get_logger().info(
                f"Selective torque {state_str}: {applied}"
                + (f" (skipped: {skipped})" if skipped else "")
            )
        res.applied_joints = applied
        return res

    def cb_get_joint_limits(self, req, res):
        """
        모든 등록된 조인트의 EEPROM Position Limit 을 rad 로 반환.

        주의:
          - hardware.read_position_limits_rad() 가 모터마다 read4ByteTxRx 두 번씩
            호출하므로 18 모터면 ~50ms 정도 걸린다. 메인 read 사이클이 잠시
            영향을 받을 수 있어 동작 중에는 호출 자제하는 게 좋다.
          - 그래서 다른 behavior 가 활성 중이면 거부.
          - 통신 실패한 모터는 응답 리스트에서 제외하고 message 에 표시.
        """
        if (self.robot_init.is_active()
                or self.standup.is_active()
                or self.kick.is_active()):
            res.success = False
            res.message = "behavior 실행 중 — 끝난 후 다시 호출"
            res.joint_names = []
            res.lower_limits_rad = []
            res.upper_limits_rad = []
            return res

        try:
            limits = self.hardware.read_position_limits_rad()
        except Exception as e:
            res.success = False
            res.message = f"read 실패: {e}"
            res.joint_names = []
            res.lower_limits_rad = []
            res.upper_limits_rad = []
            return res

        names = []
        lows = []
        ups = []
        failed = []
        for jname, info in limits.items():
            if not info["ok"]:
                failed.append(jname)
                continue
            names.append(jname)
            lows.append(float(info["lower"]))
            ups.append(float(info["upper"]))

        res.joint_names = names
        res.lower_limits_rad = lows
        res.upper_limits_rad = ups
        if not names:
            res.success = False
            res.message = f"모든 모터에서 읽기 실패: {failed}"
        else:
            res.success = True
            res.message = (
                f"limits read: {len(names)} joints"
                + (f", failed={failed}" if failed else "")
            )
            self.get_logger().info(
                f"GetJointLimits: {len(names)} joints returned"
                + (f" (failed: {failed})" if failed else "")
            )
        return res

    def _cb_grav_standup(self, msg):
        """IMU(저레이트)에서 projected gravity z 를 파생해 저장한다.

        [2026-08] 과거에는 /imu/projected_gravity(Vector3Stamped, 200Hz)를
        구독해 msg.vector.z 를 그대로 썼다. 그 토픽은 orientation 만의 순수
        함수라 중복이었고, 200Hz 구독은 executor 를 초당 200번 깨우는 비용을
        냈다. 지금은 /imu/data_50hz(sensor_msgs/Imu)를 받아 여기서 계산한다.
        값은 이전과 비트 단위로 동일하다(imu_math 주석 참조).
        """
        self.last_gravity_z = projected_gravity_from_quat(msg.orientation)[2]

    def _cb_standup(self, req, res):
        raw_cmd = (req.command or "start").strip()
        cmd = raw_cmd.lower()

        if cmd in ("start", "step", "auto"):
            if self.robot_init.is_active() or self.kick.is_active():
                res.success = False
                res.message = "robot_init/kick running"
            else:
                ok, msg = self.standup.start(
                    fall_direction=req.fall_direction or "auto",
                    gravity_z=self.last_gravity_z,
                    auto_mode=(cmd != "step"),
                    final_rpm=req.final_rpm or 200.0,
                    tol=req.pos_tolerance_rad or 0.05,
                    settle_count=req.settle_count or 5,
                    timeout_sec=req.timeout_sec or 30.0,
                )
                res.success = bool(ok)
                res.message = str(msg)

        elif cmd.startswith("goto"):
            parts = raw_cmd.split(None, 1)
            if len(parts) < 2:
                res.success = False
                res.message = "usage: 'goto <phase_name>' (e.g. 'goto base')"
            else:
                ok, msg = self.standup.goto_phase(
                    parts[1].strip(),
                    fall_direction=req.fall_direction or "forward",
                )
                res.success = bool(ok)
                res.message = str(msg)

        elif cmd == "next":
            ok, msg = self.standup.next_phase()
            res.success, res.message = bool(ok), str(msg)
        elif cmd == "back":
            ok, msg = self.standup.back_phase()
            res.success, res.message = bool(ok), str(msg)
        elif cmd == "reset":
            ok, msg = self.standup.reset_phase()
            res.success, res.message = bool(ok), str(msg)
        elif cmd == "stop":
            ok, msg = self.standup.stop()
            res.success, res.message = bool(ok), str(msg)
        elif cmd == "reload":
            ok, msg = self.standup.do_reload()
            res.success, res.message = bool(ok), str(msg)
        else:
            res.success = False
            res.message = (f"unknown: {cmd}. use: start/step/auto/next/back/reset/"
                           f"stop/goto <name>/reload")

        res.detected_direction = self.standup.get_fall_direction()
        res.current_phase = self.standup.get_phase_name()
        res.phase_index = self.standup.get_phase_idx()
        res.total_phases = self.standup.get_total_phases()
        return res

    def _cb_kick(self, req, res):
        raw_cmd = (req.command or "start").strip()
        cmd = raw_cmd.lower()
        leg = (req.leg or "right").strip().lower()

        if cmd in ("start", "auto"):
            if self.robot_init.is_active() or self.standup.is_active():
                res.success = False
                res.message = "robot_init/standup running"
            else:
                ok, msg = self.kick.start(
                    leg=leg,
                    auto_mode=(cmd == "auto"),
                    final_rpm=req.final_rpm or 200.0,
                    timeout_sec=req.timeout_sec or 10.0,
                )
                res.success = bool(ok)
                res.message = str(msg)

        elif cmd.startswith("goto"):
            parts = raw_cmd.split(None, 1)
            if len(parts) < 2:
                res.success = False
                res.message = "usage: 'goto <phase_name>' (e.g. 'goto ready')"
            else:
                ok, msg = self.kick.goto_phase(parts[1].strip(), leg=leg)
                res.success = bool(ok)
                res.message = str(msg)

        elif cmd == "next":
            ok, msg = self.kick.next_phase()
            res.success, res.message = bool(ok), str(msg)
        elif cmd == "back":
            ok, msg = self.kick.back_phase()
            res.success, res.message = bool(ok), str(msg)
        elif cmd == "reset":
            ok, msg = self.kick.reset_phase()
            res.success, res.message = bool(ok), str(msg)
        elif cmd == "stop":
            ok, msg = self.kick.stop()
            res.success, res.message = bool(ok), str(msg)
        elif cmd == "reload":
            ok, msg = self.kick.do_reload()
            res.success, res.message = bool(ok), str(msg)
        else:
            res.success = False
            res.message = (f"unknown: {cmd}. use: start/auto/next/back/reset/"
                           f"stop/goto <name>/reload")

        res.active_leg    = self.kick.get_active_leg()
        res.current_phase = self.kick.get_phase_name()
        res.phase_index   = self.kick.get_phase_idx()
        res.total_phases  = self.kick.get_total_phases()
        return res

    # ════════════════════════════════════════════════════════════════
    # Motor Reset (overload recovery)
    # ════════════════════════════════════════════════════════════════
    _ERROR_BITS = {
        0: "InputVoltage",
        2: "Overheating",
        3: "MotorEncoder",
        5: "Overload",
    }

    @classmethod
    def _describe_error(cls, code: int) -> str:
        flags = [name for bit, name in cls._ERROR_BITS.items() if code & (1 << bit)]
        return "+".join(flags) if flags else f"0x{code:02X}"

    def _cb_motor_reset(self, req, res):
        # 1) 에러 스캔
        hw_errors = self.hardware.read_hardware_errors()

        error_ids, error_codes = [], []
        for dxl_id in sorted(hw_errors.keys()):
            code = hw_errors[dxl_id]
            if code > 0:
                name = self.hardware.id_to_joint_name(dxl_id) or f"id_{dxl_id}"
                self.get_logger().warn(
                    f"Motor error: {name} (id={dxl_id}) = {self._describe_error(code)}"
                )
                error_ids.append(dxl_id)
                error_codes.append(code)

        # 2) 리부트 대상 결정
        reboot_targets = list(self.hardware.used_ids) if req.reboot_all else list(error_ids)

        rebooted = []
        for dxl_id in reboot_targets:
            comm, _err = self.hardware.reboot(dxl_id)
            if comm != 0:
                self.get_logger().warn(
                    f"Reboot fail id={dxl_id}: "
                    f"{self.hardware.packet.getTxRxResult(comm)}"
                )
            else:
                name = self.hardware.id_to_joint_name(dxl_id) or f"id_{dxl_id}"
                self.get_logger().info(f"Rebooted: {name} (id={dxl_id})")
                rebooted.append(dxl_id)

        # 3) 리부트 후 대기 + 게인 재적용 + 토크 ON
        if rebooted:
            time.sleep(0.5)

            # ★ REBOOT 는 RAM 을 초기화한다 ★
            # PID 게인이 EEPROM 기본값(P=800, I=0, D=0)으로 돌아가므로
            # 토크를 켜기 전에 반드시 다시 써야 한다. 이걸 빠뜨리면
            # 넘어졌다 일어난 뒤 로봇 거동이 달라지는데 원인을 찾을 수 없다.
            # (리부트된 모터만이 아니라 전체에 다시 쓴다 — SyncWrite 한 번이라
            #  비용이 같고, 대상을 좁히다 실수할 여지를 없앤다)
            self._apply_gains(reason="reboot")

            for dxl_id in rebooted:
                self.hardware.torque_enable(dxl_id, True)
            self.get_logger().info(
                f"Motor reset done: {len(rebooted)} rebooted, torque re-enabled"
            )

        # 4) 응답 작성
        if not error_ids and not req.reboot_all:
            res.success = True
            res.message = "no errors found"
        elif rebooted:
            res.success = True
            desc = ", ".join(
                f"{self.hardware.id_to_joint_name(d) or d}"
                f"({self._describe_error(hw_errors.get(d, 0))})"
                for d in error_ids
            ) if error_ids else "none"
            res.message = f"errors: [{desc}], rebooted {len(rebooted)} motors"
        else:
            res.success = False
            res.message = "errors found but reboot failed"

        res.error_ids    = error_ids
        res.error_codes  = error_codes
        res.rebooted_ids = rebooted
        return res

    # ════════════════════════════════════════════════════════════════
    # Behavior 가 호출하는 hw.* API surface (대부분 hardware 위임)
    # ════════════════════════════════════════════════════════════════
    def has_joint(self, joint_name: str) -> bool:
        return self.hardware.has_joint(joint_name)

    def id_to_joint_name(self, dxl_id: int):
        return self.hardware.id_to_joint_name(dxl_id)

    def send_goals(self, targets_by_name: dict):
        self.hardware.send_goals(targets_by_name)

    # ════════════════════════════════════════════════════════════════
    # PID 게인 (RAM 영역)
    # ════════════════════════════════════════════════════════════════
    def _apply_gains(self, reason: str = "") -> tuple[bool, str]:
        """게인 프로파일을 모터에 기입한다.

        호출 시점은 셋이다.
          (1) 노드 기동 직후            reason="startup"
          (2) 모터 REBOOT 직후          reason="reboot"   ← 놓치기 쉬움
          (3) reload_gains 서비스        reason="reload"

        (2) 가 특히 중요하다. REBOOT 는 RAM 을 초기화하므로 게인이 조용히
        기본값으로 돌아간다. 넘어졌다 일어난 뒤 거동이 달라지는데 원인을
        못 찾는 상황이 여기서 생긴다.

        토크는 건드리지 않는다. RAM 이라 켜진 채로 쓸 수 있고, 끄면
        로봇이 주저앉아 튜닝 반복이 불가능해진다.

        Returns:
            (성공 여부, 메시지)
        """
        if not self.gain_profile.loaded:
            # gain_file 미지정 = 게인 기능 비활성. 기존 동작 그대로.
            return True, "gain_file 미지정 → 게인 미적용"

        # 위험한 조합(P 상향 + D=0)은 쓰기 전에 경고한다.
        for w in self.gain_profile.safety_warnings(self.hardware.jid):
            self.get_logger().warn(f"[gains] {w}")

        res = self.hardware.set_pid_gains(self.gain_profile)

        # ── Goal PWM(100) 상한도 같은 경로에서 기입한다 ──
        # 게인과 완전히 같은 이유다: RAM 이라 REBOOT 시 PWM Limit(885)으로
        # 돌아간다. 낙상 복구 경로(set_torque_off_with_reset)가 에러 난 모터를
        # REBOOT 하므로, 여기서 같이 안 쓰면 넘어졌다 일어난 뒤 팔 상한이
        # 조용히 풀린다. YAML 에 goal_pwm 이 없으면 아무것도 하지 않는다.
        pwm_res = self.hardware.apply_goal_pwm_profile(self.gain_profile)

        tag = f"({reason})" if reason else ""
        if pwm_res["written"]:
            self.get_logger().info(
                f"[pwm] Goal PWM 상한 적용 {tag}: {pwm_res['written']}개 관절")

        if res["failed"]:
            msg = (f"게인 적용 일부 실패 {tag}: "
                   f"written={res['written']} failed={res['failed']}")
            self.get_logger().warn(f"[gains] {msg}")
            return False, msg

        msg = f"게인 적용 {tag} {res['written']}개 | " \
              f"{self.gain_profile.summary(self.hardware.jid)}"
        self.get_logger().info(f"[gains] {msg}")

        # 검증: 실제 레지스터와 대조.
        # 기동/재적용 시에만 하므로 비용은 무시할 수준이다.
        if bool(self.get_parameter("gain_verify").value):
            bad = self.hardware.verify_pid_gains(self.gain_profile)
            if bad:
                for b in bad:
                    self.get_logger().warn(f"[gains] 검증 불일치: {b}")
                return False, msg + f" (검증 불일치 {len(bad)}건)"

        return True, msg

    def _cb_reload_gains(self, req, res):
        """YAML 을 다시 읽어 재적용한다. 튜닝 반복용.

        노드를 재시작하지 않고, 토크도 끄지 않는다.
        로봇이 걷는 중에도 호출해 게인을 바꿔가며 비교할 수 있다.
        """
        path = str(self.get_parameter("gain_file").value)
        if not path:
            res.success = False
            res.message = "gain_file 파라미터가 비어 있습니다."
            return res

        if not self.gain_profile.load(path):
            res.success = False
            res.message = f"YAML 로드 실패: {path}"
            return res

        ok, msg = self._apply_gains(reason="reload")
        res.success = ok
        res.message = msg
        return res

    def set_torque_all(self, enable: bool):
        self.hardware.set_torque_all(enable)
        self._publish_torque_enabled(enable, 'set_torque_all')

    def set_profile_velocity_all(self, rpm: float):
        self.hardware.set_profile_velocity_all(rpm)

    def set_profile_velocity_joints(self, rpm_by_joint: dict) -> list:
        """관절별 개별 Profile Velocity (behavior 가 호출하는 hw.* API).

        RobotInitBehavior 가 목만 느린 속도(≈10rpm)로 보내는 데 사용한다.
        일괄 쓰기에서 제외된 관절도 이 경로로는 정상 기입된다.
        """
        return self.hardware.set_profile_velocity_joints(rpm_by_joint)

    def set_goal_pwm_full(self, joint_names=None):
        """behavior 가 호출: 지정 관절(기본 전체)의 Goal PWM 을 100% 로 올린다.

        robot_init / standup 처럼 팔로 실제 힘을 써야 하는 동작 직전에 부른다.
        상한이 낮게 걸린 채로는 팔이 목표까지 이동하지 못해 도달 판정이
        timeout 으로 끝난다(실측: 1.0rad 이동에 최소 150 필요, 유지는 30).
        """
        return self.hardware.set_goal_pwm_full(joint_names)

    def restore_goal_pwm(self):
        """behavior 가 호출: 게인 프로파일의 goal_pwm 상한으로 되돌린다.

        set_goal_pwm_full() 로 올린 것을 반드시 여기서 내린다. 안 내리면
        팔이 100% 상한인 채로 보행에 들어가 낙상 시 감속기 보호가 사라진다.
        """
        return self.hardware.apply_goal_pwm_profile(self.gain_profile)

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def log_info(self, s: str):
        self.get_logger().info(s)

    def log_warn(self, s: str):
        self.get_logger().warn(s)

    # ════════════════════════════════════════════════════════════════
    # Lifecycle
    # ════════════════════════════════════════════════════════════════
    def destroy_node(self):
        try:
            self.hardware.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BodyMotorController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
