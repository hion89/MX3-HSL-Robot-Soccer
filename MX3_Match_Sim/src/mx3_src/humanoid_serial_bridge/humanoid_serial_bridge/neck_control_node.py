#!/usr/bin/env python3
"""
neck_control_node.py
====================

목(Neck) deg 단위 API  ↔  U2D2 관절 제어(rad) 어댑터.

────────────────────────────────────────────────────────────────────────────
[2026-08 이관] AX-18A(MCU 버스) → MX-64(U2D2 버스)

  기존:
    /neck/yaw_deg → neck_control_node → /mcu/raw_tx → STM32 MCU → AX-18A(ID19/20)
                                        /mcu/raw_rx → neck_state_parser_node
                                                    → /joint_states

  현재:
    /neck/yaw_deg → neck_control_node → /joint_goals → body_motor_controller
                                                     → U2D2 → MX-64(ID19/20)
    /neck/*_deg_state ←──────────────── /joint_states ←─────────┘

  즉 이 노드는 더 이상 시리얼 프레임을 만들지 않는다. 하는 일은 딱 두 가지:
    (1) deg 명령  → URDF 관절 rad 로 바꿔 /joint_goals 에 실어 보내기
    (2) /joint_states 의 목 관절 rad → deg 로 되돌려 상태 토픽으로 재발행
        (삭제된 neck_state_parser_node 의 역할을 흡수)

  덕분에 상위 노드(strategy_fsm_node, neck_ball_tracker_node, neck_scan_node)는
  토픽 이름도 부호 규약도 그대로여서 한 줄도 고치지 않았다.
────────────────────────────────────────────────────────────────────────────

[부호 규약] — 기존 AX-18A 시절과 100% 동일하게 유지
    API deg:
        yaw   + = 왼쪽 (REP-103 yaw. 반시계방향이 양수), - = 오른쪽
        pitch + = 위,  - = 아래. 정면이 0.
    URDF 관절 rad (= /joint_states, /joint_goals 에 실리는 값):
        neck_yaw_joint   =  radians(yaw_deg)      (yaw_sign  = +1)
        neck_pitch_joint = -radians(pitch_deg)    (pitch_sign = -1)
    ★ pitch 는 API 와 URDF 의 부호가 반대다. 기존 neck_state_parser_node 의
      기본 파라미터(pitch_sign=-1.0)를 그대로 옮겨온 것이라, TF/URDF 쪽은
      아무것도 손대지 않아도 된다.

    이 노드가 담당하는 부호   : API deg  ↔  URDF 관절 rad   (규약 변환)
    servo_map 이 담당하는 부호 : URDF 관절 rad ↔ 모터 tick   (기구 장착 방향)
    두 개를 헷갈리지 말 것. 부호가 반대로 나오면 어느 쪽인지부터 구분해야 한다.

[단위 변경 주의]
    MOVING_SPEED(AX raw, 0.111 rpm/unit) → Profile Velocity(rpm)
    위치 raw (AX 0~1023, center 512)     → MX 0~4095, center 2048
    /neck/moving_speed 는 neck_scan_node 하위호환을 위해 AX raw 로 계속 받고
    내부에서 rpm 으로 환산한다 (raw 90 ≈ 10 rpm — 기존과 동일한 속도).

Subscribed topics:
    /neck/yaw_deg          (Float32)            yaw 목표 [deg]
    /neck/pitch_deg        (Float32)            pitch 목표 [deg]
    /neck/yaw_position     (UInt16)             yaw 목표 [MX raw tick 0..4095]
    /neck/pitch_position   (UInt16)             pitch 목표 [MX raw tick]
    /neck/yaw_pitch        (UInt16MultiArray)   [yaw_tick, pitch_tick] 동시 지정
    /neck/moving_speed     (UInt16)             속도 [AX raw, 하위호환]
    /neck/speed_rpm        (Float32)            속도 [rpm, 신규 권장]
    /neck/torque_enable    (Bool)               목 2관절 토크 ON/OFF
    /neck/init_cmd         (Float32MultiArray)  [yaw_deg, pitch_deg, rpm, torque]
    <joint_states_topic>   (JointState)         현재 관절값 수신 (상태 환산용)

Published topics:
    <goal_topic>              (JointState)          기본 /joint_goals
    <profile_velocity_topic>  (JointState)          기본 /u2d2/set_joint_profile_velocity
    /neck/yaw_deg_state       (Float32)             현재 yaw   [deg]
    /neck/pitch_deg_state     (Float32)             현재 pitch [deg]
    /neck/cmd_deg             (Float32MultiArray)   디버그: 마지막 명령 [yaw, pitch] deg

Service clients:
    /u2d2/torque_enable_joints (TorqueEnableJoints)  목만 토크 ON/OFF
"""
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import (
    Bool, Float32, Float32MultiArray, UInt16, UInt16MultiArray,
)

from humanoid_behavior_interfaces.srv import TorqueEnableJoints


# ── MX-64 / XM540 공통 (Protocol 2.0) ────────────────────────────────────
# 분해능 4096 tick / 360°, 중앙 2048. AX-18A(1024/300°, 512) 와 다르다.
MX_CENTER_TICK   = 2048
MX_TICK_MAX      = 4095
MX_DEG_PER_TICK  = 360.0 / 4096.0     # = 0.087890625 °/tick

# AX-18A MOVING_SPEED raw → rpm 환산 (하위호환 경로 전용).
# neck_scan_node 와 launch 의 scan_speed_raw(기본 90)가 아직 이 단위를 쓴다.
#   raw 90 × 0.111 ≈ 10 rpm  → 이관 전과 같은 속도가 나온다.
AX_RAW_TO_RPM = 0.111


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class NeckControlNode(Node):
    def __init__(self):
        super().__init__('neck_control_node')

        # ── 관절명 / 토픽 (URDF·컨트롤러와 맞춰야 함) ──────────────
        self.declare_parameter('yaw_joint_name',   'neck_yaw_joint')
        self.declare_parameter('pitch_joint_name', 'neck_pitch_joint')
        self.declare_parameter('goal_topic',         '/joint_goals')
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('profile_velocity_topic',
                               '/u2d2/set_joint_profile_velocity')
        self.declare_parameter('torque_service', '/u2d2/torque_enable_joints')

        self._yaw_name   = str(self.get_parameter('yaw_joint_name').value)
        self._pitch_name = str(self.get_parameter('pitch_joint_name').value)
        goal_topic       = str(self.get_parameter('goal_topic').value)
        js_topic         = str(self.get_parameter('joint_states_topic').value)
        pv_topic         = str(self.get_parameter('profile_velocity_topic').value)
        torque_srv       = str(self.get_parameter('torque_service').value)

        # ── 기구 안전 한계 (deg, API 규약) ─────────────────────────
        #   YAW  : 좌우 ±90°  (+ = 왼쪽)
        #   PITCH: 정면(0°) ~ 아래(-60°)  (+ = 위)
        #   실제 최종 방어선은 모터 EEPROM 의 Min/Max Position Limit 이다.
        #   여기 clamp 는 그 앞단의 소프트 방어선.
        self.declare_parameter('yaw_deg_min',   -90.0)
        self.declare_parameter('yaw_deg_max',    90.0)
        self.declare_parameter('pitch_deg_min', -60.0)
        self.declare_parameter('pitch_deg_max',   0.0)
        self._yaw_min   = float(self.get_parameter('yaw_deg_min').value)
        self._yaw_max   = float(self.get_parameter('yaw_deg_max').value)
        self._pitch_min = float(self.get_parameter('pitch_deg_min').value)
        self._pitch_max = float(self.get_parameter('pitch_deg_max').value)

        # ── 부호 / 영점 오프셋 (구 neck_state_parser_node 규약 계승) ──
        #   rad = radians((deg - offset) * sign)
        #   deg = degrees(rad) / sign + offset
        self.declare_parameter('yaw_sign',    1.0)
        self.declare_parameter('pitch_sign', -1.0)
        self.declare_parameter('yaw_offset_deg',   0.0)
        self.declare_parameter('pitch_offset_deg', 0.0)
        self._yaw_sign   = float(self.get_parameter('yaw_sign').value)
        self._pitch_sign = float(self.get_parameter('pitch_sign').value)
        self._yaw_off    = float(self.get_parameter('yaw_offset_deg').value)
        self._pitch_off  = float(self.get_parameter('pitch_offset_deg').value)
        # 부호가 0 이면 나눗셈에서 터진다. ±1 로 강제.
        self._yaw_sign   = 1.0 if self._yaw_sign   >= 0.0 else -1.0
        self._pitch_sign = 1.0 if self._pitch_sign >= 0.0 else -1.0

        # ── 속도 (Profile Velocity, rpm) ───────────────────────────
        #   기본값 10rpm ≈ 60°/s. 카메라 흔들림과 인식 안정성의 절충값.
        #   ★ 0 은 Dynamixel 에서 "무제한(최대속도)" 이라 절대 내려보내면 안 된다.
        #     아래 _publish_speed 에서 하한을 강제한다.
        self.declare_parameter('default_speed_rpm', 10.0)
        self.declare_parameter('speed_rpm_min', 0.5)
        self.declare_parameter('speed_rpm_max', 30.0)
        self._default_rpm = float(self.get_parameter('default_speed_rpm').value)
        self._rpm_min     = float(self.get_parameter('speed_rpm_min').value)
        self._rpm_max     = float(self.get_parameter('speed_rpm_max').value)

        # init_cmd 시퀀스에서 '속도 기입 → 목표 발행' 사이에 두는 간격(초).
        #   다이나믹셀은 Goal Position 을 쓰는 그 순간의 Profile Velocity 로
        #   이동을 시작한다. 속도 메시지(/u2d2/set_joint_profile_velocity)와
        #   목표 메시지(/joint_goals)는 body_motor_controller 의 서로 다른
        #   콜백/타이머에서 처리되므로, 붙여서 쏘면 목표가 먼저 버스로 나가
        #   직전 속도로 출발할 수 있다. 한 사이클(20ms)보다 넉넉히 띄운다.
        #   init 은 1회성이라 50ms 지연은 체감되지 않는다.
        self.declare_parameter('init_step_delay_sec', 0.05)
        self._init_delay = float(self.get_parameter('init_step_delay_sec').value)
        self._init_timer = None      # one-shot 타이머 핸들
        self._init_goal  = None      # 지연 후 보낼 (yaw_deg, pitch_deg)

        # ── ROS I/O ────────────────────────────────────────────────
        self.pub_goal  = self.create_publisher(JointState, goal_topic, 10)
        self.pub_speed = self.create_publisher(JointState, pv_topic, 10)
        self.pub_yaw_state   = self.create_publisher(
            Float32, '/neck/yaw_deg_state', 10)
        self.pub_pitch_state = self.create_publisher(
            Float32, '/neck/pitch_deg_state', 10)
        # 디버그용: 마지막으로 명령한 각도. 상태 토픽과 같이 플롯하면
        # 추종 지연/오버슛을 눈으로 바로 확인할 수 있다.
        self.pub_cmd_deg = self.create_publisher(
            Float32MultiArray, '/neck/cmd_deg', 10)

        # deg 명령
        self.create_subscription(Float32, '/neck/yaw_deg',   self._on_yaw_deg, 10)
        self.create_subscription(Float32, '/neck/pitch_deg', self._on_pitch_deg, 10)
        # raw tick 명령 (수동 테스트용. 단위가 AX→MX 로 바뀌었으니 주의)
        self.create_subscription(UInt16, '/neck/yaw_position',
                                 self._on_yaw_tick, 10)
        self.create_subscription(UInt16, '/neck/pitch_position',
                                 self._on_pitch_tick, 10)
        self.create_subscription(UInt16MultiArray, '/neck/yaw_pitch',
                                 self._on_yaw_pitch_tick, 10)
        # 속도
        self.create_subscription(UInt16, '/neck/moving_speed',
                                 self._on_speed_ax_raw, 10)
        self.create_subscription(Float32, '/neck/speed_rpm',
                                 self._on_speed_rpm, 10)
        # 토크
        self.create_subscription(Bool, '/neck/torque_enable', self._on_torque, 10)
        # 결합 init 명령
        self.create_subscription(Float32MultiArray, '/neck/init_cmd',
                                 self._on_init_cmd, 10)
        # 현재 상태 (rad → deg 역변환용)
        self.create_subscription(JointState, js_topic, self._on_joint_states, 20)

        self.cli_torque = self.create_client(TorqueEnableJoints, torque_srv)

        # ── 상태 캐시 / 통계 ───────────────────────────────────────
        self._last_yaw_deg   = None    # 명령한 값
        self._last_pitch_deg = None
        self._cur_yaw_deg    = None    # 실제 측정값
        self._cur_pitch_deg  = None
        self._state_count    = 0       # 1초 창의 상태 수신 횟수
        self._speed_primed   = False   # 기동 후 기본 속도를 한 번 기입했는가
        self._warned_tick_unit = False

        self.create_timer(1.0, self._print_status)

        self.get_logger().info(
            f'neck_control_node (U2D2 어댑터) 시작 — '
            f'{self._yaw_name}/{self._pitch_name} → {goal_topic}, '
            f'yaw=[{self._yaw_min:+.0f}°..{self._yaw_max:+.0f}°] '
            f'pitch=[{self._pitch_min:+.0f}°..{self._pitch_max:+.0f}°], '
            f'sign(yaw={self._yaw_sign:+.0f}, pitch={self._pitch_sign:+.0f}), '
            f'기본속도 {self._default_rpm:.1f} rpm')

    # ═══════════════════════════════════════════════════════════════
    # 단위 변환
    # ═══════════════════════════════════════════════════════════════
    def _yaw_deg_to_rad(self, deg: float) -> float:
        return math.radians((deg - self._yaw_off) * self._yaw_sign)

    def _pitch_deg_to_rad(self, deg: float) -> float:
        return math.radians((deg - self._pitch_off) * self._pitch_sign)

    def _yaw_rad_to_deg(self, rad: float) -> float:
        return math.degrees(rad) / self._yaw_sign + self._yaw_off

    def _pitch_rad_to_deg(self, rad: float) -> float:
        return math.degrees(rad) / self._pitch_sign + self._pitch_off

    def _tick_to_deg(self, tick: int) -> float:
        """MX raw tick → deg (중앙 2048 기준). 부호/오프셋 보정은 하지 않는다.

        raw tick 명령은 '모터 눈금 그대로' 주고 싶을 때 쓰는 수동 테스트용
        경로라, API deg 규약이 아니라 모터 눈금 기준으로 해석한다.
        """
        return (int(tick) - MX_CENTER_TICK) * MX_DEG_PER_TICK

    # ═══════════════════════════════════════════════════════════════
    # 발행 헬퍼
    # ═══════════════════════════════════════════════════════════════
    def _send_goal(self, yaw_deg=None, pitch_deg=None):
        """지정된 관절만 /joint_goals 로 발행한다.

        한쪽만 보내도 되는 이유:
          body_motor_controller.cb_goals 가 '관절 단위 병합' 이라, 이번에
          안 실은 관절은 직전 목표를 그대로 유지한다. 매번 두 관절을 다
          싣지 않아도 되고, 정책이 보내는 다리 18관절을 덮어쓰지도 않는다.
          (컨트롤러가 구버전이면 여기서 다리 명령이 지워지므로,
           body_motor_controller 도 반드시 같이 빌드할 것)
        """
        names, positions = [], []

        if yaw_deg is not None:
            d = clamp(float(yaw_deg), self._yaw_min, self._yaw_max)
            names.append(self._yaw_name)
            positions.append(self._yaw_deg_to_rad(d))
            self._last_yaw_deg = d

        if pitch_deg is not None:
            d = clamp(float(pitch_deg), self._pitch_min, self._pitch_max)
            names.append(self._pitch_name)
            positions.append(self._pitch_deg_to_rad(d))
            self._last_pitch_deg = d

        if not names:
            return

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = names
        js.position = positions
        self.pub_goal.publish(js)

        # 디버그 토픽: 명령값 [yaw, pitch].
        # 아직 한 번도 명령되지 않은 축은 NaN 으로 둔다. 0.0 을 넣으면
        # "정면을 명령했다" 와 구분이 안 돼서 플롯 해석이 틀어진다.
        def _or_nan(v):
            return float(v) if v is not None else float('nan')

        dbg = Float32MultiArray()
        dbg.data = [_or_nan(self._last_yaw_deg), _or_nan(self._last_pitch_deg)]
        self.pub_cmd_deg.publish(dbg)

    def _publish_speed(self, rpm: float, reason: str = ""):
        """목 2관절의 Profile Velocity 를 rpm 단위로 기입 요청한다.

        0 은 다이나믹셀에서 '무제한(=Velocity Limit 최대속도)' 이라 절대
        내려보내면 안 된다. 최소값으로 강제 clamp 한다.
        """
        v = clamp(float(rpm), self._rpm_min, self._rpm_max)
        if abs(v - float(rpm)) > 1e-6:
            self.get_logger().warn(
                f'목 속도 {float(rpm):.2f} rpm → {v:.2f} rpm 으로 clamp '
                f'(허용 {self._rpm_min}~{self._rpm_max} rpm)')

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = [self._yaw_name, self._pitch_name]
        js.velocity = [v, v]
        self.pub_speed.publish(js)
        self.get_logger().info(
            f'목 Profile Velocity {v:.1f} rpm 요청' + (f' ({reason})' if reason else ''))

    # ═══════════════════════════════════════════════════════════════
    # 명령 콜백
    # ═══════════════════════════════════════════════════════════════
    def _on_yaw_deg(self, msg: Float32):
        if not math.isfinite(msg.data):
            self.get_logger().warn('/neck/yaw_deg 에 NaN/inf 수신 → 무시')
            return
        self._send_goal(yaw_deg=msg.data)

    def _on_pitch_deg(self, msg: Float32):
        if not math.isfinite(msg.data):
            self.get_logger().warn('/neck/pitch_deg 에 NaN/inf 수신 → 무시')
            return
        self._send_goal(pitch_deg=msg.data)

    def _warn_tick_unit_once(self):
        if self._warned_tick_unit:
            return
        self._warned_tick_unit = True
        self.get_logger().warn(
            'raw tick 명령 수신 — 단위가 AX-18A(0~1023, center 512) 에서 '
            'MX-64(0~4095, center 2048) 로 바뀌었습니다. '
            '예전 값을 그대로 쓰면 엉뚱한 각도로 갑니다.')

    def _on_yaw_tick(self, msg: UInt16):
        self._warn_tick_unit_once()
        self._send_goal(yaw_deg=self._tick_to_deg(clamp(msg.data, 0, MX_TICK_MAX)))

    def _on_pitch_tick(self, msg: UInt16):
        self._warn_tick_unit_once()
        self._send_goal(pitch_deg=self._tick_to_deg(clamp(msg.data, 0, MX_TICK_MAX)))

    def _on_yaw_pitch_tick(self, msg: UInt16MultiArray):
        if len(msg.data) < 2:
            self.get_logger().warn('/neck/yaw_pitch 는 [yaw_tick, pitch_tick] 2개가 필요합니다')
            return
        self._warn_tick_unit_once()
        self._send_goal(
            yaw_deg=self._tick_to_deg(clamp(msg.data[0], 0, MX_TICK_MAX)),
            pitch_deg=self._tick_to_deg(clamp(msg.data[1], 0, MX_TICK_MAX)),
        )

    def _on_speed_ax_raw(self, msg: UInt16):
        """구 AX MOVING_SPEED raw 로 들어오는 속도 (neck_scan_node 하위호환).

        raw 0 은 AX 에서도 '최대속도' 라 그대로 넘기면 안 된다.
        _publish_speed 의 clamp 가 최종적으로 막아준다.
        """
        self._publish_speed(int(msg.data) * AX_RAW_TO_RPM,
                            reason=f'AX raw {int(msg.data)} 환산')

    def _on_speed_rpm(self, msg: Float32):
        self._publish_speed(msg.data, reason='/neck/speed_rpm')

    def _on_torque(self, msg: Bool):
        """목 2관절만 토크 ON/OFF (/u2d2/torque_enable_joints 서비스 호출).

        전신 토크(/u2d2/torque_enable)는 목까지 자동으로 덮으므로, 이 경로는
        '목만' 켜고 끌 때 쓴다 (neck_scan 시작, 부호 검증 등).
        """
        if not self.cli_torque.service_is_ready():
            self.get_logger().warn(
                '토크 서비스가 아직 준비되지 않음 → 목 토크 명령 무시 '
                '(body_motor_controller 기동 확인)')
            return
        req = TorqueEnableJoints.Request()
        req.joint_names = [self._yaw_name, self._pitch_name]
        req.enable = bool(msg.data)
        # 응답을 기다리지 않는다. 콜백 안에서 spin 하면 단일 스레드
        # executor 가 그대로 데드락에 걸린다.
        future = self.cli_torque.call_async(req)
        future.add_done_callback(self._on_torque_done)
        self.get_logger().info(f"목 토크 {'ON' if msg.data else 'OFF'} 요청")

    def _on_torque_done(self, future):
        try:
            res = future.result()
        except Exception as e:
            self.get_logger().warn(f'목 토크 서비스 실패: {e}')
            return
        if not res.success:
            # behavior 실행 중이면 컨트롤러가 거부한다. 원인을 그대로 노출.
            self.get_logger().warn(f'목 토크 거부됨: {res.message}')

    def _on_init_cmd(self, msg: Float32MultiArray):
        """init/zero 용 결합 명령. '토크 → 속도 → (지연) → 목표' 순으로 처리.

        data 레이아웃 (길이 4):
            [0] yaw_deg, [1] pitch_deg
            [2] speed_rpm   ★ 단위 변경: AX raw → rpm
            [3] torque      : >0 = ON, ==0 = OFF, <0 = 건드리지 않음

        [왜 지연을 두는가]
          다이나믹셀은 Goal Position 을 쓰는 그 순간의 Profile Velocity 로
          이동을 시작한다. 속도와 목표는 서로 다른 토픽 → 다른 콜백/타이머로
          처리되므로 붙여 쏘면 목표가 먼저 나갈 수 있고, 그러면 직전 속도로
          출발한다. init_step_delay_sec 만큼 띄워 순서를 보장한다.
        """
        if len(msg.data) < 4:
            self.get_logger().warn(
                '/neck/init_cmd 는 [yaw_deg, pitch_deg, speed_rpm, torque] 4개가 필요합니다')
            return

        yaw_deg   = float(msg.data[0])
        pitch_deg = float(msg.data[1])
        speed_rpm = float(msg.data[2])
        torque    = float(msg.data[3])

        # 1) 토크
        if torque > 0.5:
            self._on_torque(Bool(data=True))
        elif torque < -0.5:
            pass                       # 건드리지 않음
        else:
            self._on_torque(Bool(data=False))

        # 2) 속도
        self._publish_speed(speed_rpm, reason='init_cmd')

        # 3) 목표 (지연 후)
        self._init_goal = (yaw_deg, pitch_deg)
        if self._init_timer is not None:
            self.destroy_timer(self._init_timer)
            self._init_timer = None
        if self._init_delay <= 0.0:
            self._fire_init_goal()
        else:
            self._init_timer = self.create_timer(self._init_delay, self._fire_init_goal)

    def _fire_init_goal(self):
        """init_cmd 의 목표를 지연 후 1회 발행 (반복 타이머를 one-shot 으로 사용)."""
        if self._init_timer is not None:
            self._init_timer.cancel()
            self.destroy_timer(self._init_timer)
            self._init_timer = None
        if self._init_goal is None:
            return
        yaw_deg, pitch_deg = self._init_goal
        self._init_goal = None
        self._send_goal(yaw_deg=yaw_deg, pitch_deg=pitch_deg)

    # ═══════════════════════════════════════════════════════════════
    # 상태 수신 (/joint_states → deg 재발행)
    # ═══════════════════════════════════════════════════════════════
    def _on_joint_states(self, msg: JointState):
        """body_motor_controller 가 SyncRead 로 읽은 목 관절 rad → deg 로 역변환.

        삭제된 neck_state_parser_node 의 역할을 그대로 흡수한 부분이다.
        neck_ball_tracker_node 가 /neck/*_deg_state 를 구독하므로, 이 발행이
        끊기면 시선 추적이 통째로 멈춘다.

        /joint_states 에는 다리/팔도 함께 실려 오므로 목이 없는 메시지는
        조용히 무시한다(다른 퍼블리셔가 있을 수도 있다).
        """
        got = False
        for name, pos in zip(msg.name, msg.position):
            if name == self._yaw_name:
                self._cur_yaw_deg = self._yaw_rad_to_deg(pos)
                self.pub_yaw_state.publish(Float32(data=float(self._cur_yaw_deg)))
                got = True
            elif name == self._pitch_name:
                self._cur_pitch_deg = self._pitch_rad_to_deg(pos)
                self.pub_pitch_state.publish(Float32(data=float(self._cur_pitch_deg)))
                got = True

        if not got:
            return
        self._state_count += 1

        # 첫 상태 수신 = body_motor_controller 가 살아 있다는 확실한 신호.
        # 이 시점에 기본 속도를 한 번 기입해 둔다. 아무도 속도를 안 정해주면
        # 모터에 남아 있던 직전 값(전원 투입 직후엔 사실상 최대속도)으로
        # 움직여 카메라가 크게 흔들린다.
        if not self._speed_primed:
            self._speed_primed = True
            self._publish_speed(self._default_rpm, reason='기동 시 기본값')

    # ═══════════════════════════════════════════════════════════════
    # 1Hz 상태 로그
    # ═══════════════════════════════════════════════════════════════
    def _print_status(self):
        hz = self._state_count
        self._state_count = 0

        if hz == 0:
            self.get_logger().warn(
                '목 상태 수신 없음 — /joint_states 에 목 관절이 없습니다. '
                'servo_map.yaml 의 joint_map 에 neck_yaw_joint/neck_pitch_joint '
                '가 있는지, 모터 ID/보레이트가 맞는지 확인하세요.')
            return

        def fmt(v):
            return f'{v:+.1f}°' if v is not None else '--'

        self.get_logger().info(
            f'neck {hz:d}Hz | '
            f'cur yaw={fmt(self._cur_yaw_deg)} pitch={fmt(self._cur_pitch_deg)} | '
            f'cmd yaw={fmt(self._last_yaw_deg)} pitch={fmt(self._last_pitch_deg)}')


def main(args=None):
    rclpy.init(args=args)
    node = NeckControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Ctrl+C는 정상 종료 흐름. 스택트레이스 안 찍음.
        pass
    finally:
        node.destroy_node()
        # 시그널 핸들러가 이미 shutdown 했을 수 있으니 가드
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
