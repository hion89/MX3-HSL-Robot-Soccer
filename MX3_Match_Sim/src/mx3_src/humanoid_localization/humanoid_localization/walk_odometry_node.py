#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  walk_odometry_node.py
#
#  KHUBER 휴머노이드용 "보행 오도메트리(walk odometry)" 노드.
#
#  [핵심 아이디어]
#  --------------------------------------------------------------------------
#  바퀴형 로봇처럼 엔코더로 거리를 잴 수 없는 보행 로봇에서는,
#  "보행 정책에 내려준 명령 속도(cmd_vel)"를 실제 속도로 환산(calibration)한 뒤
#  시간으로 적분하여 위치를 추정한다. 이를 open-loop dead reckoning 이라 부른다.
#
#    실제속도 v_real = f(명령속도 v_cmd)        ← calibration (아래 lookup table)
#    위치 x,y       = ∫ (v_real 을 yaw 로 회전시킨 값) dt
#    heading yaw    = IMU 의 yaw (gyro 적분 기반 AHRS 값)
#
#  [중요한 한계 — 반드시 인지할 것]
#  --------------------------------------------------------------------------
#   1) 이 오도메트리는 "마커 기반 localization 을 대체"하지 않는다.
#      표준 TF 구조는 다음과 같다:
#          map  --(마커 localization)-->  odom  --(이 노드)-->  base_link
#      즉 이 노드는 odom->base_link 만 책임지는 "단기 추측 항법"이다.
#      마커가 보일 때마다 상위(localization) 노드가 map->odom 으로 누적오차를 보정한다.
#
#   2) cmd_vel -> v_real 매핑(아래 _CALIB_*)은 측정 데이터로 채워야 정확하다.
#      현재 기본값은 단 2점(vx=0.3->0.235, vx=0.4->0.286 m/s)에서 얻은 임시값이며,
#      이 2점에는 "출발 가속 구간"이 포함되어 정속(steady-state)보다 낮게 측정된
#      값이다. 더 많은 속도(0.1/0.2/0.3/0.4 ...)에서, 그리고 가속 구간을 제외한
#      구간(예: 1m 통과~5m 통과)을 재측정하여 테이블을 보강할 것을 강력히 권장한다.
#
#   3) vy(횡이동) 는 별도로 측정/보정해야 한다. 직진과 횡이동의 실제/명령 비율은 다르다.
#      현재는 단일 스케일(_VY_SCALE)로 근사한다. 측정 전이라면 보수적으로 둔다.
#
#   4) 회전(wz)은 명령값을 적분하지 않고 IMU yaw 를 직접 쓴다(권장).
#      6축 IMU(LSM6DSV16X, 자력계 없음)라서 yaw 는 gyro 적분 → 장시간 드리프트가 있으나
#      한 번의 set-play/접근 구간 정도의 단기 추정에는 충분하다.
#
#  [입력 토픽]
#     /cmd_vel  (geometry_msgs/Twist) : 정책에 내리는 명령 속도(보행 의도)
#     /imu/data (sensor_msgs/Imu)     : orientation(yaw 추출) + angular_velocity.z(gyro)
#
#  [출력]
#     /odom     (nav_msgs/Odometry)   : Nav2 가 구독하는 표준 오도메트리
#     TF        odom -> base_link     : (publish_tf=True 일 때) 좌표변환 브로드캐스트
# =============================================================================

import math

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import (
    Twist, Quaternion, TransformStamped, PoseWithCovarianceStamped)
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
import rclpy.time


# =============================================================================
#  Calibration 기본 테이블 (cmd vx -> 실제 vx)
# -----------------------------------------------------------------------------
#  np.interp 로 선형 보간한다. 측정점을 늘릴수록 정확해진다.
#  주의: 측정점은 반드시 "오름차순"으로 정렬되어 있어야 한다(np.interp 요구사항).
#  범위를 벗어난 명령은 np.interp 가 자동으로 양끝 값으로 clamp 한다(외삽하지 않음).
#  → 이는 의도된 보수적 동작이다. 검증되지 않은 속도에서 위험하게 외삽하지 않는다.
#
#  현재 측정 데이터:
#     [전진] vx 0.1 → 정지(데드존), 0.2 → 움직이기 시작, 0.3 → 4/17=0.2353, 0.4 → 4/14=0.2857
#     [후진] vx 0.2 까지 정지(데드존), 0.3 → 움직이기 시작
#  ★ 테이블에 후진(음수)~전진(양수) 전 구간을 직접 담는다. 이렇게 하면:
#     - 데드존: 작은 명령에서 real=0 (보간이 0 을 내놓음)
#     - 비대칭: 전/후진 데드존·기울기가 달라도 테이블 그대로 반영
#    부호를 따로 복원할 필요가 없다. calib_vx_cmd 는 반드시 오름차순(음수→양수).
#    아직 미측정인 칸(0.2 전진 실제속도, 0.3/0.4 후진 실제속도)은 추정값이며,
#    측정 후 odom_calib.yaml 만 갱신하면 된다.
# =============================================================================
_CALIB_VX_CMD  = [-0.4,  -0.3,  -0.2, -0.1, 0.0, 0.1, 0.2,  0.3,    0.4]
_CALIB_VX_REAL = [-0.18, -0.02,  0.0,  0.0, 0.0, 0.0, 0.05, 0.2353, 0.2857]

# vy(횡이동) 보정 스케일. 직진과 동특성이 다르므로 별도 측정이 원칙이다.
# 미측정 상태에서는 직진 평균 스케일(~0.75)보다 더 보수적으로 두는 편이 안전하다.
_VY_SCALE = 0.6


def yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
    """쿼터니언(x,y,z,w) 에서 yaw(=Z축 회전, rad) 만 추출한다.

    표준 ZYX(roll-pitch-yaw) 분해식 중 yaw 항만 사용:
        yaw = atan2( 2(qw*qz + qx*qy),  1 - 2(qy^2 + qz^2) )
    roll/pitch 가 작을 때 정확하며, 보행 로봇의 평면 heading 추정에 충분하다.
    """
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float) -> Quaternion:
    """yaw(rad) 만 갖는(roll=pitch=0) 평면 쿼터니언을 생성한다.

    odom 의 자세는 평면 heading 만 의미 있으므로 yaw 만으로 쿼터니언을 만든다:
        qz = sin(yaw/2),  qw = cos(yaw/2)
    """
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


class WalkOdometryNode(Node):
    """명령속도 + IMU yaw 기반 보행 오도메트리 노드."""

    def __init__(self):
        super().__init__('walk_odometry_node')

        # ---------------------------------------------------------------------
        #  파라미터 선언 (ros2 param / launch 파일에서 덮어쓰기 가능)
        # ---------------------------------------------------------------------
        # calibration 테이블: 실측으로 늘려가며 런타임/런치에서 교체할 수 있게 노출한다.
        self.declare_parameter('calib_vx_cmd',  _CALIB_VX_CMD)
        self.declare_parameter('calib_vx_real', _CALIB_VX_REAL)
        self.declare_parameter('vy_scale',      _VY_SCALE)

        # 적분 주기(Hz). 정책/IMU 갱신율보다 빠를 필요는 없으나, dt 가 너무 크면
        # 회전 중 적분 오차가 커지므로 50Hz 정도를 기본으로 한다.
        self.declare_parameter('publish_rate', 50.0)

        # 프레임 이름 (Nav2/TF 트리 규약)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')

        # odom->base_link TF 를 직접 쏠지 여부.
        # 다른 노드(예: ekf/robot_localization)가 TF 를 책임진다면 False 로 둔다.
        self.declare_parameter('publish_tf', True)

        # heading 소스로 IMU yaw 를 쓸지 여부.
        # False 면 명령 wz 를 적분(드리프트는 작지만 정확도는 IMU 보다 낮음).
        self.declare_parameter('use_imu_yaw', True)
        self.declare_parameter('game_stop_lock_topic', 'game_stop_lock')
        self.declare_parameter('freeze_on_game_stop_lock', True)
        self.declare_parameter('cmd_vel_timeout_sec', 0.3)
        self.declare_parameter('cmd_vel_active_epsilon', 1.0e-4)
        self.declare_parameter('kick_done_topic', '/kick_done')
        self.declare_parameter('post_kick_imu_yaw_sec', 3.0)

        # odom->base_link TF 의 z 오프셋(m).
        #   walk_odometry 는 2D(z=0) 라 base_link 을 지면 평면에 두는데,
        #   base_link 은 골반 중심이고 다리가 그 아래로 뻗으므로 RViz 에서
        #   발이 지면 아래로 박힌다. base_link 을 직립 시 높이만큼 띄워
        #   발바닥이 z=0 바닥에 놓이게 한다(시각화용).
        #   값 = 직립 시 base_link 에서 발바닥까지의 수직 거리.
        #   측정: ros2 run tf2_ros tf2_echo base_link l_ankle_roll_link 의 z(절댓값) + 발두께
        #   주의: /odom 토픽 z 는 0 으로 유지(Nav2/2D 평면). TF 만 띄운다.
        self.declare_parameter('base_height', 0.0)

        # 파라미터 읽어서 멤버로 보관
        self._calib_cmd  = np.array(
            self.get_parameter('calib_vx_cmd').value, dtype=float)
        self._calib_real = np.array(
            self.get_parameter('calib_vx_real').value, dtype=float)
        self._vy_scale   = float(self.get_parameter('vy_scale').value)
        self._rate       = float(self.get_parameter('publish_rate').value)
        self._odom_frame = self.get_parameter('odom_frame').value
        self._base_frame = self.get_parameter('base_frame').value
        self._publish_tf = bool(self.get_parameter('publish_tf').value)
        self._use_imu_yaw = bool(self.get_parameter('use_imu_yaw').value)
        self._game_stop_lock_topic = self.get_parameter(
            'game_stop_lock_topic').value
        self._freeze_on_game_stop_lock = bool(
            self.get_parameter('freeze_on_game_stop_lock').value)
        self._cmd_vel_timeout = float(
            self.get_parameter('cmd_vel_timeout_sec').value)
        self._cmd_vel_active_epsilon = float(
            self.get_parameter('cmd_vel_active_epsilon').value)
        self._kick_done_topic = self.get_parameter('kick_done_topic').value
        self._post_kick_imu_yaw_sec = max(
            0.0, float(self.get_parameter('post_kick_imu_yaw_sec').value))
        self._base_height = float(self.get_parameter('base_height').value)

        # calibration 테이블 유효성 검사: np.interp 는 x 가 오름차순이어야 함.
        if not np.all(np.diff(self._calib_cmd) > 0):
            self.get_logger().error(
                'calib_vx_cmd 가 오름차순이 아닙니다 — 보간 결과가 부정확해집니다.')

        # ---------------------------------------------------------------------
        #  적분 상태(state). odom 프레임 기준 누적값.
        # ---------------------------------------------------------------------
        self._x = 0.0          # odom 기준 x 위치 [m]
        self._y = 0.0          # odom 기준 y 위치 [m]
        self._yaw = 0.0        # odom 기준 heading [rad] (시작점 = 0)

        # 가장 최근에 받은 명령 속도(body frame). cmd_vel 콜백에서 갱신.
        self._cmd_vx = 0.0     # 전진 명령 [m/s]
        self._cmd_vy = 0.0     # 횡이동 명령 [m/s]
        self._cmd_wz = 0.0     # 회전 명령 [rad/s] (use_imu_yaw=False 일 때만 사용)
        self._cmd_time = None

        # IMU 관련 상태
        self._imu_yaw_raw = None   # IMU 가 주는 절대 yaw(rad). 첫 수신 전엔 None.
        self._yaw_offset = None    # 노드 시작 시점 yaw 를 0 으로 맞추기 위한 보정값.
        self._gyro_z = 0.0         # gyro z [rad/s] (odom twist.angular.z 에 사용)
        self._imu_odom_active = False
        self._game_stop_locked = False
        self._odom_frozen = False
        self._post_kick_imu_until = None

        # 적분 dt 계산용 직전 타임스탬프(초). 첫 tick 에서 초기화.
        self._last_time = None

        # ---------------------------------------------------------------------
        #  QoS: 센서/명령 스트림이므로 best-effort + 작은 큐로 지연을 줄인다.
        #  (IMU/cmd 는 최신값만 의미 있으므로 신뢰성보다 최신성을 우선)
        # ---------------------------------------------------------------------
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ---------------------------------------------------------------------
        #  구독 / 발행 / TF
        # ---------------------------------------------------------------------
        self.create_subscription(Twist, '/cmd_vel', self._cb_cmd_vel, 10)

        # ── [2026-08 CPU 최적화] IMU 구독을 저레이트 채널로 전환 ───────────
        #   _cb_imu 는 절대 yaw 와 gyro_z 를 '저장만' 하고, 실제 사용은
        #   self._timer(publish_rate=50Hz)에서 일어난다. 적분도 여기서 안 한다.
        #   즉 200Hz 로 받아도 4개 중 3개는 깨어나서 버렸다.
        #   rclpy 는 깨어날 때마다 wait set 을 재구축하므로 그게 그대로 CPU 다.
        #   (실측: 이 노드 37.5% 중 상당 부분)
        #
        #   지연 영향: Madgwick 의 '절대' yaw 를 샘플링만 하므로, 값이 최대
        #   15ms 더 오래된 것이 된다. 보행속도 0.3m/s 기준 위치 오차 약 4.5mm.
        #   적분 오차가 누적되는 구조가 아니라 단순 지연이므로 무시 가능하다.
        #
        #   되돌리려면 launch 에서 imu_topic:=/imu/data 로 지정하면 된다.
        self.declare_parameter('imu_topic', '/imu/data_50hz')
        self._imu_topic = str(self.get_parameter('imu_topic').value)
        self.create_subscription(Imu, self._imu_topic, self._cb_imu, sensor_qos)
        self.create_subscription(
            Bool, self._game_stop_lock_topic, self._cb_game_stop_lock, 10)
        self.create_subscription(
            Bool, self._kick_done_topic, self._cb_kick_done, 10)

        # RViz "2D Pose Estimate"(/initialpose) 로 오도메트리 시작점을 런타임에 지정.
        #   - localization(AMCL 등)이 없으므로 이 노드가 직접 받아 적분 상태를 리셋한다.
        #   - 받은 pose 는 보통 map 프레임 → tf2 로 odom 프레임으로 변환해 반영.
        self.create_subscription(
            PoseWithCovarianceStamped, '/initialpose', self._cb_initialpose, 10)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self._tf_broadcaster = TransformBroadcaster(self) if self._publish_tf else None

        # 일정 주기로 적분 + 발행
        self._timer = self.create_timer(1.0 / self._rate, self._on_timer)

        self.get_logger().info(
            f'walk_odometry 시작: rate={self._rate}Hz, '
            f'imu_topic={self._imu_topic}, '
            f'use_imu_yaw={self._use_imu_yaw}, publish_tf={self._publish_tf}')
        self.get_logger().info(
            f'game_stop_lock freeze={self._freeze_on_game_stop_lock}, '
            f'topic="{self._game_stop_lock_topic}"')
        self.get_logger().info(
            f'calib vx_cmd={self._calib_cmd.tolist()} -> '
            f'vx_real={self._calib_real.tolist()}, vy_scale={self._vy_scale}')

    # =========================================================================
    #  콜백
    # =========================================================================
    def _cb_cmd_vel(self, msg: Twist):
        """정책에 내려간 명령 속도를 저장한다(body frame).

        actor_onnx_runner 의 매핑과 동일:
            linear.x  = vx (전진/후진)
            linear.y  = vy (좌/우 횡이동)
            angular.z = wz (회전)
        """
        self._cmd_vx = float(msg.linear.x)
        self._cmd_vy = float(msg.linear.y)
        self._cmd_wz = float(msg.angular.z)
        self._cmd_time = self.get_clock().now().nanoseconds * 1e-9

    def _cb_imu(self, msg: Imu):
        """IMU 에서 yaw(절대) 와 gyro z 를 추출해 저장한다."""
        q = msg.orientation
        self._imu_yaw_raw = yaw_from_quaternion(q.x, q.y, q.z, q.w)

        # 노드 시작 후 첫 IMU 수신 시점을 heading 0 기준으로 삼는다.
        # → odom 프레임의 원점에서 로봇이 +x 를 향하도록 정의(표준 관례).
        if self._yaw_offset is None:
            self._yaw_offset = self._imu_yaw_raw
            self.get_logger().info(
                f'IMU yaw 기준점 설정: offset={math.degrees(self._yaw_offset):.2f} deg')

        # twist.angular.z 보고용 gyro z (rad/s)
        self._gyro_z = float(msg.angular_velocity.z)

    def _cb_kick_done(self, msg: Bool):
        if not bool(msg.data) or self._post_kick_imu_yaw_sec <= 0.0:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        self._post_kick_imu_until = now + self._post_kick_imu_yaw_sec
        self.get_logger().info(
            f'kick_done -> IMU yaw accepted without cmd for '
            f'{self._post_kick_imu_yaw_sec:.1f}s')

    def _post_kick_imu_active(self, now: float) -> bool:
        if self._post_kick_imu_until is None:
            return False
        if now <= self._post_kick_imu_until:
            return True
        self._post_kick_imu_until = None
        return False

    def _cb_game_stop_lock(self, msg: Bool):
        """game_stop_lock=True 동안 odom 적분과 IMU yaw 반영을 멈춘다."""
        locked = bool(msg.data)
        if locked == self._game_stop_locked:
            return

        self._game_stop_locked = locked
        if not self._freeze_on_game_stop_lock:
            return

        if locked:
            self._odom_frozen = True
            self._cmd_vx = 0.0
            self._cmd_vy = 0.0
            self._cmd_wz = 0.0
            self._cmd_time = None
            self._imu_odom_active = False
            self.get_logger().warn(
                'game_stop_lock ON -> odom freeze '
                f'(x={self._x:.3f}, y={self._y:.3f}, '
                f'yaw={math.degrees(self._yaw):.1f}deg)')
            return

        if self._odom_frozen:
            if self._use_imu_yaw and self._imu_yaw_raw is not None:
                # Discard yaw drift accumulated while frozen and resume from
                # the held odom yaw without a discontinuity.
                self._yaw_offset = math.atan2(
                    math.sin(self._imu_yaw_raw - self._yaw),
                    math.cos(self._imu_yaw_raw - self._yaw),
                )
            self._imu_odom_active = False
            self._odom_frozen = False
            self._last_time = self.get_clock().now().nanoseconds * 1e-9
            self.get_logger().warn(
                'game_stop_lock OFF -> odom resume '
                f'(x={self._x:.3f}, y={self._y:.3f}, '
                f'yaw={math.degrees(self._yaw):.1f}deg)')

    def _cb_initialpose(self, msg: PoseWithCovarianceStamped):
        """RViz 2D Pose Estimate 로 지정한 pose 로 오도메트리 상태를 리셋한다.

        동작:
          1) 받은 pose 의 프레임(보통 'map')에서 odom 프레임으로 tf 변환을 조회.
          2) 2D 변환식으로 목표 (x, y, yaw) 를 odom 프레임 값으로 계산.
          3) 위치는 그대로 세팅. yaw 는 use_imu_yaw 인 경우 'IMU 기준점(offset)'을
             다시 잡아, 다음 tick 부터 지정한 yaw 가 나오도록 한다.
             (self._yaw 를 직접 쓰지 않고 offset 을 고쳐야 IMU 와 어긋나지 않음)
        """
        src = msg.header.frame_id or 'map'
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        src_yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)

        # src 프레임 → odom 프레임 변환 조회. 같은 프레임이면 항등으로 처리.
        tx = ty = t_yaw = 0.0
        if src != self._odom_frame:
            try:
                tf = self._tf_buffer.lookup_transform(
                    self._odom_frame, src, rclpy.time.Time())
                tx = tf.transform.translation.x
                ty = tf.transform.translation.y
                tq = tf.transform.rotation
                t_yaw = yaw_from_quaternion(tq.x, tq.y, tq.z, tq.w)
            except Exception as e:
                self.get_logger().warn(
                    f'initialpose: {src}->{self._odom_frame} tf 조회 실패({e}). '
                    f'동일 프레임으로 가정하고 진행.')

        # 2D 강체변환: p_odom = T_translation + R(t_yaw) * p_src
        c, s = math.cos(t_yaw), math.sin(t_yaw)
        self._x = tx + c * px - s * py
        self._y = ty + s * px + c * py
        target_yaw = math.atan2(math.sin(t_yaw + src_yaw),
                                math.cos(t_yaw + src_yaw))

        if self._use_imu_yaw and self._imu_yaw_raw is not None:
            # 다음 tick 에서 self._yaw = imu_yaw_raw - offset = target_yaw 가 되도록 offset 갱신
            self._yaw_offset = math.atan2(
                math.sin(self._imu_yaw_raw - target_yaw),
                math.cos(self._imu_yaw_raw - target_yaw))
        self._yaw = target_yaw
        self._imu_odom_active = False

        self.get_logger().info(
            f'initialpose 적용: x={self._x:.3f}, y={self._y:.3f}, '
            f'yaw={math.degrees(target_yaw):.1f}deg (src="{src}")')

    # =========================================================================
    #  calibration: 명령 vx -> 실제 vx
    # =========================================================================
    def _calibrate_vx(self, vx_cmd: float) -> float:
        """명령 전진속도를 실측 lookup table 로 실제 속도로 환산한다.

        테이블(calib_vx_cmd/real)이 후진(음수)~전진(양수) 전 구간을 직접 담으므로
        부호 처리 없이 그대로 보간한다. 이로써 다음이 자연스럽게 표현된다:
          - 데드존: 작은 명령(전진 <0.2, 후진 >-0.3 부근)에서 real≈0 → 로봇 정지로 추정
          - 비대칭: 전/후진 데드존·기울기 차이를 테이블이 그대로 반영
        테이블 범위 밖은 np.interp 가 양끝값으로 clamp(외삽 금지).
        ★ calib_vx_cmd 는 반드시 오름차순(음수→양수)이어야 한다.
        """
        return float(np.interp(vx_cmd, self._calib_cmd, self._calib_real))

    def _cmd_vel_fresh(self, now: float) -> bool:
        if self._cmd_time is None:
            return False
        return (now - self._cmd_time) <= self._cmd_vel_timeout

    def _cmd_vel_active(self, now: float) -> bool:
        if not self._cmd_vel_fresh(now):
            return False
        eps = self._cmd_vel_active_epsilon
        return (abs(self._cmd_vx) > eps
                or abs(self._cmd_vy) > eps
                or abs(self._cmd_wz) > eps)

    # =========================================================================
    #  메인 적분 루프
    # =========================================================================
    def _on_timer(self):
        # --- dt 계산 (ROS 시계 기준) ---
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._last_time is None:
            # 첫 tick 은 dt 를 알 수 없으므로 적분하지 않고 시각만 기록한다.
            self._last_time = now
            return
        dt = now - self._last_time
        self._last_time = now

        # 비정상 dt(시계 점프 등) 방어: 0 이하이거나 비현실적으로 크면 이번 스텝은 건너뛴다.
        if dt <= 0.0 or dt > 0.5:
            return

        if self._odom_frozen:
            stamp = self.get_clock().now().to_msg()
            self._publish_odom(stamp, 0.0, 0.0, 0.0)
            if self._tf_broadcaster is not None:
                self._publish_tf_msg(stamp)
            return

        cmd_fresh = self._cmd_vel_fresh(now)
        cmd_active = self._cmd_vel_active(now)
        post_kick_imu_active = self._post_kick_imu_active(now)
        if not cmd_fresh:
            self._cmd_vx = 0.0
            self._cmd_vy = 0.0
            self._cmd_wz = 0.0
            if not post_kick_imu_active:
                self._imu_odom_active = False

        # --- 1) 명령속도 -> 실제 body-frame 속도로 환산 ---
        if cmd_active:
            vx_body = self._calibrate_vx(self._cmd_vx)   # 전진 실제속도 [m/s]
            vy_body = self._cmd_vy * self._vy_scale       # 횡이동 실제속도 [m/s]
        else:
            vx_body = 0.0
            vy_body = 0.0
            if not post_kick_imu_active:
                self._imu_odom_active = False

        # --- 2) heading(yaw) 갱신 ---
        if ((cmd_active or post_kick_imu_active)
                and self._use_imu_yaw
                and self._imu_yaw_raw is not None):
            if self._yaw_offset is None:
                self._yaw_offset = self._imu_yaw_raw
            if cmd_active and not post_kick_imu_active and not self._imu_odom_active:
                self._yaw_offset = math.atan2(
                    math.sin(self._imu_yaw_raw - self._yaw),
                    math.cos(self._imu_yaw_raw - self._yaw),
                )
                self._imu_odom_active = True
            # IMU 절대 yaw 에서 시작 offset 을 빼 odom 기준 상대 heading 으로.
            # atan2 로 [-pi, pi] 정규화(연속성 유지).
            self._yaw = math.atan2(
                math.sin(self._imu_yaw_raw - self._yaw_offset),
                math.cos(self._imu_yaw_raw - self._yaw_offset),
            )
            self._imu_odom_active = True
        elif cmd_active:
            self._imu_odom_active = True
            # IMU 미사용/미수신 시: 명령 wz 를 적분(폴백). 정확도는 낮다.
            self._yaw += self._cmd_wz * dt
            self._yaw = math.atan2(math.sin(self._yaw), math.cos(self._yaw))

        # --- 3) body-frame 속도를 odom(world) 프레임으로 회전시켜 위치 적분 ---
        #   [dx]   [cos(yaw)  -sin(yaw)] [vx_body]
        #   [dy] = [sin(yaw)   cos(yaw)] [vy_body] * dt
        c, s = math.cos(self._yaw), math.sin(self._yaw)
        self._x += (c * vx_body - s * vy_body) * dt
        self._y += (s * vx_body + c * vy_body) * dt

        # --- 4) 발행 ---
        stamp = self.get_clock().now().to_msg()
        if ((cmd_active or post_kick_imu_active)
                and self._use_imu_yaw
                and self._imu_yaw_raw is not None):
            wz_body = self._gyro_z
        elif cmd_active:
            wz_body = self._cmd_wz
        else:
            wz_body = 0.0
        self._publish_odom(stamp, vx_body, vy_body, wz_body)
        if self._tf_broadcaster is not None:
            self._publish_tf_msg(stamp)

    # =========================================================================
    #  발행 헬퍼
    # =========================================================================
    def _publish_odom(self, stamp, vx_body: float, vy_body: float, wz_body=None):
        """nav_msgs/Odometry 메시지를 채워 발행한다."""
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self._odom_frame   # 위치/자세가 표현된 기준 프레임
        odom.child_frame_id = self._base_frame    # twist(속도)가 표현된 프레임(=로봇 몸체)

        # pose: odom 기준 위치/자세
        odom.pose.pose.position.x = self._x
        odom.pose.pose.position.y = self._y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = quaternion_from_yaw(self._yaw)

        # twist: child_frame(base_link) 기준 속도
        odom.twist.twist.linear.x = vx_body
        odom.twist.twist.linear.y = vy_body
        odom.twist.twist.angular.z = (
            self._gyro_z if wz_body is None else float(wz_body))

        # 공분산: open-loop 추정이라 신뢰도가 낮음을 Nav2/EKF 에 알린다.
        # 대각 성분만 채우고 나머지는 0. 값이 클수록 "덜 믿는다"는 뜻.
        # (실측 분산을 얻으면 이 값들을 데이터 기반으로 교체할 것)
        odom.pose.covariance[0]  = 0.05   # x
        odom.pose.covariance[7]  = 0.05   # y
        odom.pose.covariance[35] = 0.02   # yaw (IMU 사용 시 비교적 신뢰)
        odom.twist.covariance[0]  = 0.05  # vx
        odom.twist.covariance[7]  = 0.10  # vy (보정 불확실 → 더 크게)
        odom.twist.covariance[35] = 0.02  # wz

        self._odom_pub.publish(odom)

    def _publish_tf_msg(self, stamp):
        """odom -> base_link TF 를 브로드캐스트한다."""
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self._odom_frame
        t.child_frame_id = self._base_frame
        t.transform.translation.x = self._x
        t.transform.translation.y = self._y
        t.transform.translation.z = self._base_height   # 시각화용 base 높이(odom 토픽은 z=0 유지)
        t.transform.rotation = quaternion_from_yaw(self._yaw)
        self._tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = WalkOdometryNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # Ctrl+C / 외부 shutdown — 정상 종료 경로
        pass
    except RuntimeError:
        # 종료 시퀀스 중 executor 가 마지막 구독 메시지를 역직렬화하다 나는 race
        # ("Unable to convert call argument..."). context 가 이미 내려가는
        # 중이면(rclpy.ok()==False) 무해하므로 삼키고, 동작 중(rclpy.ok()==True)
        # 에 난 거라면 진짜 에러이므로 그대로 재발생시킨다.
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        # 시그널 핸들러가 이미 shutdown 했다면 다시 호출하지 않는다
        # (rcl_shutdown already called 방지).
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
