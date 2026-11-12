#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  relocalize_button_node.py
# -----------------------------------------------------------------------------
#  역할
#    SW1(좌/우 토글) + 실행버튼(/buttons/start_event)으로 "재진입 시작 위치"를
#    자동으로 로컬라이저에 시드한다.
#
#    기존엔 RViz '2D Pose Estimate'(=/initialpose)를 마우스로 찍어 위치를
#    지정했는데, 그걸 하드웨어 입력으로 자동화한다. 로봇이 퇴장당하거나
#    고장으로 빠졌다가 재진입할 때, 정해진 사이드라인 바깥 자리에 로봇을 세우고
#    실행버튼을 누르면 그 자리로 시드가 리셋된다.
#
#  흐름
#    /board/dip/sw1     (std_msgs/Bool, 토글)  → 최신 좌/우 상태를 캐시
#    /buttons/start_event (std_msgs/Empty)     → 누르는 "순간"(엣지)에:
#         sideline_reentry_pose(geom, sw1) 로 (x, y, yaw) 계산
#         → /initialpose (PoseWithCovarianceStamped, map 프레임) 발행
#    field_landmark_localizer_node 가 이 /initialpose 를 받아
#    다음 N 프레임 동안 "강제 시드"로 재수렴한다(재진입 복구용 경로 그대로 재사용).
#
#  좌표/헤딩 규약 (field_geometry 와 동일)
#    원점=필드 중앙, +x=길이(상대 골 쪽), +y=폭(왼쪽), 단위 m/rad.
#    - 우리 진영은 항상 -x (own_goal_sign=-1 고정). → SW2 불필요.
#    - SW1=False → 왼쪽(+y) 사이드라인 바깥, SW1=True → 오른쪽(-y) 사이드라인 바깥.
#    - 시작점: 우리 골라인 근처(x), 사이드라인 바깥(|y|>half_width).
#    - 헤딩: 필드 안쪽(Y축 평행). 왼쪽이면 -90°, 오른쪽이면 +90°.
#
#  설계 메모
#    - field_profile 은 launch 인자로 localizer 와 "같은 값"을 받아야 한다.
#      (시드 좌표가 localizer 의 랜드마크 지도와 같은 스케일이어야 하므로)
#    - SW1 은 ~10Hz 로 계속 스트리밍되므로 토글 상태가 항상 최신이다.
#      실행버튼은 부팅 후 누르는 것이라(노드 이미 떠 있음) /initialpose 의
#      QoS/타이밍 핸드셰이크 걱정이 없다(엣지 트리거).
#    - 실행버튼은 kickoff 에도 눌리는데, kickoff 때도 로봇은 시작 위치에 있으니
#      재시드해도 올바르다. 경기 중 오발 우려가 있으면 STOP 상태에서만 시드하도록
#      가드를 추가할 수 있다(현재는 단순히 매 start_event 마다 시드).
# =============================================================================

import math

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException

from std_msgs.msg import Bool, Empty
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion

from .field_geometry import field_geometry_for_profile, sideline_reentry_pose


def _quaternion_from_yaw(yaw: float) -> Quaternion:
    """yaw(rad)만 갖는 평면 쿼터니언(roll=pitch=0)."""
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


def _pose_covariance(pos_var: float, yaw_var: float):
    """PoseWithCovariance 6x6(36) 공분산(행우선).

    2D(x, y, yaw)만 의미 있으므로 z/roll/pitch 에는 큰 값(1e6)을 넣어
    EKF 가 무시하게 한다. 대각 인덱스: x=0, y=7, z=14, roll=21, pitch=28, yaw=35.
    """
    cov = [0.0] * 36
    big = 1e6
    cov[0] = pos_var    # x
    cov[7] = pos_var    # y
    cov[14] = big       # z      (미사용)
    cov[21] = big       # roll   (미사용)
    cov[28] = big       # pitch  (미사용)
    cov[35] = yaw_var   # yaw
    return cov


class RelocalizeButtonNode(Node):
    def __init__(self):
        super().__init__('relocalize_button_node')

        # ---------------------------------------------------------------------
        #  파라미터
        # ---------------------------------------------------------------------
        self.declare_parameter('field_profile', 'practice')        # localizer 와 동일하게
        self.declare_parameter('sw1_topic', '/board/dip/sw1')
        self.declare_parameter('start_event_topic', '/buttons/start_event')
        self.declare_parameter('initialpose_topic', '/initialpose')
        self.declare_parameter('map_frame', 'map')

        # 사이드라인 재진입 기하 (sideline_reentry_pose 인자)
        self.declare_parameter('own_goal_sign', -1)        # 우리 골 = -x (고정)
        self.declare_parameter('sw1_false_is_left', True)  # SW1=False → 왼쪽(+y)
        self.declare_parameter('outside_margin_m', 0.4)    # 사이드라인 바깥 여유(m)
        self.declare_parameter('x_inset_m', 0.0)           # 골라인에서 안쪽으로 당김(m)
        # 헤딩: 기본은 자동(필드 안쪽 ±90°). 물리적으로 다르게 세울 때만 override.
        self.declare_parameter('override_yaw', False)
        self.declare_parameter('start_yaw_deg', 0.0)       # override_yaw=True 일 때 사용

        # 시드 확신도(표준편차 → 분산). 작을수록 강하게 시드.
        self.declare_parameter('seed_pos_std_m', 0.10)
        self.declare_parameter('seed_yaw_std_deg', 5.0)

        # SW1 메시지를 아직 못 받았을 때 가정할 기본값(안전망)
        self.declare_parameter('default_sw1', False)

        gp = self.get_parameter
        profile = gp('field_profile').value
        # 잘못된 프로파일이면 field_geometry_for_profile 가 명확히 에러를 낸다.
        self._geom = field_geometry_for_profile(profile)

        self._map_frame = gp('map_frame').value
        self._own_goal_sign = int(gp('own_goal_sign').value)
        self._sw1_false_is_left = bool(gp('sw1_false_is_left').value)
        self._outside_margin_m = float(gp('outside_margin_m').value)
        self._x_inset_m = float(gp('x_inset_m').value)
        self._override_yaw = bool(gp('override_yaw').value)
        self._start_yaw_deg = float(gp('start_yaw_deg').value)
        self._pos_var = float(gp('seed_pos_std_m').value) ** 2
        self._yaw_var = math.radians(float(gp('seed_yaw_std_deg').value)) ** 2

        self._sw1 = bool(gp('default_sw1').value)
        self._sw1_received = False

        # ---------------------------------------------------------------------
        #  퍼블리셔 / 구독
        # ---------------------------------------------------------------------
        self._pub = self.create_publisher(
            PoseWithCovarianceStamped, gp('initialpose_topic').value, 10)
        self.create_subscription(
            Bool, gp('sw1_topic').value, self._cb_sw1, 10)
        self.create_subscription(
            Empty, gp('start_event_topic').value, self._cb_start, 10)

        self.get_logger().info(
            f"relocalize_button_node 시작. profile='{profile}' "
            f"(field {self._geom.length_m:.2f}x{self._geom.width_m:.2f}m). "
            f"구독: SW1='{gp('sw1_topic').value}', "
            f"START='{gp('start_event_topic').value}' "
            f"→ 발행 '{gp('initialpose_topic').value}'. "
            f"실행버튼 누르면 SW1 위치로 시드한다.")

    # =========================================================================
    #  콜백
    # =========================================================================
    def _cb_sw1(self, msg: Bool):
        """SW1 토글 상태를 최신값으로 캐시(변경 시 로그)."""
        new = bool(msg.data)
        if new != self._sw1 or not self._sw1_received:
            self.get_logger().info(
                f"SW1 = {new} ({self._side_name(new)})")
        self._sw1 = new
        self._sw1_received = True

    def _cb_start(self, _msg: Empty):
        """실행버튼(엣지) → 현재 SW1 위치로 /initialpose 발행."""
        start_yaw = self._start_yaw_deg if self._override_yaw else None
        x, y, yaw = sideline_reentry_pose(
            self._geom, self._sw1,
            own_goal_sign=self._own_goal_sign,
            sw1_false_is_left=self._sw1_false_is_left,
            outside_margin_m=self._outside_margin_m,
            x_inset_m=self._x_inset_m,
            start_yaw_deg=start_yaw,
        )

        out = PoseWithCovarianceStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self._map_frame      # map 프레임 → localizer 가 변환 없이 사용
        out.pose.pose.position.x = x
        out.pose.pose.position.y = y
        out.pose.pose.position.z = 0.0
        out.pose.pose.orientation = _quaternion_from_yaw(yaw)
        out.pose.covariance = _pose_covariance(self._pos_var, self._yaw_var)
        self._pub.publish(out)

        warn = '' if self._sw1_received else '  ※ SW1 미수신 → default_sw1 사용'
        self.get_logger().info(
            f"실행버튼 → 재시드: SW1={self._sw1}({self._side_name(self._sw1)}), "
            f"x={x:.3f}, y={y:.3f}, yaw={math.degrees(yaw):.1f}deg{warn}")

    # =========================================================================
    #  헬퍼
    # =========================================================================
    def _side_name(self, sw1: bool) -> str:
        is_left = (not sw1) if self._sw1_false_is_left else sw1
        return '왼쪽 +y' if is_left else '오른쪽 -y'


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RelocalizeButtonNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
