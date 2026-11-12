#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  field_lines_node.py
#
#  RoboCup 경기장의 흰 라인(+골 기둥)을 RViz 에 그려주는 노드.
#
#  [치수 출처 — 단일화]  (단위 m, 필드 정중앙이 원점, +x=골대 방향, +y=왼쪽)
#    경기장 치수는 yaml 에 숫자로 두지 않고 field_profile 로부터 field_geometry
#    (vision/field_geometry.py)에서 "생성"한다. 이는 localizer/relocalize_button 이
#    쓰는 바로 그 프로파일과 동일하므로, 라인 시각화와 측위 랜드마크가 같은 좌표를
#    공유한다(중복 숫자로 인한 drift 원천 차단).
#       field_profile:=practice  → 연습장 3.9 x 2.6 (인조 잔디 5x3 안의 라인)
#       field_profile:=s_field   → 대회 9 x 6 (HSL 2026 small division; competition 별칭)
#
#    yaml(config/field.yaml)에는 "치수가 아닌" 시각화 전용값만 남는다:
#       선 두께, 골 기둥 높이/지름, 잔디 외곽 표시, 발행 주기 등.
#
#  [표현 방식] 벽이 아니므로 OccupancyGrid 가 아니라 visualization_msgs/Marker
#             (LINE_LIST/LINE_STRIP)로 라인을 직접 그린다.
#
#  [한계] 라인은 시각 참고용일 뿐 충돌/내비게이션에 영향을 주지 않는다.
#         로봇의 절대 위치 보정(localization)은 별도 노드가 담당한다.
# =============================================================================

import math

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

# 경기장 "치수"의 단일 출처. localizer/relocalize_button 과 동일한 프로파일을 써서
# RViz 라인과 측위 랜드마크가 절대 어긋나지 않게 한다(중복 숫자 제거).
from .vision.field_geometry import field_geometry_for_profile


def _pt(x: float, y: float, z: float = 0.0) -> Point:
    """(x, y, z) → geometry_msgs/Point 단축 헬퍼."""
    p = Point()
    p.x = float(x); p.y = float(y); p.z = float(z)
    return p


class FieldLinesNode(Node):
    def __init__(self):
        super().__init__('field_lines_node')

        # ---------------------------------------------------------------------
        #  파라미터
        #  --------------------------------------------------------------------
        #  [핵심 변경] 경기장 "치수"는 더 이상 yaml 에 숫자로 직접 두지 않는다.
        #  localizer / relocalize_button 과 똑같이 field_profile(연습장/대회)에서
        #  field_geometry 로 "생성"한다 → 단일 출처(single source of truth)가 되어
        #  RViz 라인과 측위 랜드마크가 절대 어긋나지 않는다.
        #     field_profile:=practice  → 연습장 3.9 x 2.6
        #     field_profile:=s_field   → 대회 9 x 6 (= "competition" 별칭)
        #  → vision_localization.launch.py 가 localizer 와 "같은" field_profile 을
        #    이 노드에도 주입하므로, 런치 인자 한 개로 전부 같이 전환된다.
        #
        #  yaml 에는 "치수가 아닌" 시각화 전용값(선 두께/골 기둥 높이·지름/잔디/주기)만
        #  남긴다. 골대 폭(goal_width)은 치수이므로 프로파일에서 온다(yaml 아님).
        # ---------------------------------------------------------------------
        self.declare_parameter('field_profile', 'practice')   # localizer 와 동일하게 주입
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('line_thickness', 0.05)        # 라인 굵기(viz; 규정 0.05)
        self.declare_parameter('draw_goals', True)            # 골 기둥 표시 여부
        self.declare_parameter('goal_post_height', 1.2)       # 골 기둥 높이(viz; S-Field 1.2~1.3)
        self.declare_parameter('goal_post_radius', 0.04)      # 기둥 반지름(viz; 규정 지름 7~12cm)
        self.declare_parameter('draw_turf_boundary', True)    # 연습장 잔디 외곽(5x3) 표시
        self.declare_parameter('turf_length', 5.0)            # 연습장 잔디 가로(viz 전용)
        self.declare_parameter('turf_width', 3.0)             # 연습장 잔디 세로(viz 전용)
        self.declare_parameter('publish_rate', 1.0)

        g = self.get_parameter

        # --- 경기장 치수: 프로파일에서 생성(단일 출처) ---
        profile = str(g('field_profile').value)
        # 잘못된 프로파일이면 field_geometry_for_profile 가 명확히 에러를 낸다(조용한 오동작 금지).
        geom = field_geometry_for_profile(profile)
        self._L  = geom.length_m                   # 외곽 라인 가로(x)
        self._W  = geom.width_m                    # 외곽 라인 세로(y)
        self._ccr = geom.center_circle_radius_m    # 센터 서클 반지름
        self._pad = geom.penalty_area_depth_m      # 페널티 에어리어 깊이
        self._paw = geom.penalty_area_width_m      # 페널티 에어리어 폭
        self._gad = geom.goal_area_depth_m         # 골 에어리어 깊이
        self._gaw = geom.goal_area_width_m         # 골 에어리어 폭
        self._pmd = geom.penalty_mark_distance_from_goal_line_m  # 골라인→페널티 마크
        self._pms = geom.spot_cross_half_size_m * 2.0  # 마크 십자 전체 길이(geom 은 half 보관)
        self._gw  = geom.goal_width_m              # 골대 두 기둥 간격(치수 → 프로파일)

        # --- 시각화 전용값: yaml ---
        self._frame = g('frame_id').value
        self._lt = float(g('line_thickness').value)
        self._draw_goals = bool(g('draw_goals').value)
        self._gph = float(g('goal_post_height').value)
        self._gpr = float(g('goal_post_radius').value)
        # 잔디 외곽(5x3)은 연습장 도면 전용이라, 대회(9x6) 프로파일에선 자동으로 끈다.
        #   (켜두면 큰 필드 안에 엉뚱한 5x3 박스가 그려져 혼동을 준다.)
        self._draw_turf = bool(g('draw_turf_boundary').value) and (profile == 'practice')
        self._tL = float(g('turf_length').value)
        self._tW = float(g('turf_width').value)
        rate = float(g('publish_rate').value)

        # latched QoS (늦게 켠 RViz 도 수신). RViz 측 Durability 도 Transient Local 로.
        qos = QoSProfile(depth=1,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self._pub = self.create_publisher(MarkerArray, '/field_markers', qos)

        self._markers = self._build_markers()
        self._pub.publish(self._markers)
        self.create_timer(1.0 / max(rate, 0.1), self._tick)

        self.get_logger().info(
            f'field_lines: line {self._L}x{self._W}m, turf {self._tL}x{self._tW}m, '
            f'frame="{self._frame}"')

    def _tick(self):
        now = self.get_clock().now().to_msg()
        for m in self._markers.markers:
            m.header.stamp = now
        self._pub.publish(self._markers)

    # =========================================================================
    def _line_marker(self, ns, mid, color):
        m = Marker()
        m.header.frame_id = self._frame
        m.ns = ns; m.id = mid
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.scale.x = self._lt
        m.color = color
        m.pose.orientation.w = 1.0
        return m

    def _build_markers(self) -> MarkerArray:
        white = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        arr = MarkerArray()

        L2, W2 = self._L / 2.0, self._W / 2.0

        # ---- (1) 흰 라인 LINE_LIST ----
        lines = self._line_marker('field_lines', 0, white)

        def seg(x1, y1, x2, y2):
            lines.points.append(_pt(x1, y1))
            lines.points.append(_pt(x2, y2))

        # 외곽 경계 직사각형
        seg(-L2, -W2,  L2, -W2)
        seg(-L2,  W2,  L2,  W2)
        seg(-L2, -W2, -L2,  W2)
        seg( L2, -W2,  L2,  W2)

        # 센터 라인
        seg(0.0, -W2, 0.0, W2)

        def goal_box(side_x, depth, half_w):
            """골라인(side_x) 앞 ㄷ자 박스(골라인 변 제외 3변)."""
            inner = side_x + depth if side_x < 0 else side_x - depth
            seg(side_x, -half_w, inner, -half_w)
            seg(side_x,  half_w, inner,  half_w)
            seg(inner, -half_w, inner,  half_w)

        # 페널티 에어리어(큰) + 골 에어리어(작은), 양쪽. 깊이/폭은 프로파일에서 옴.
        for sx in (-L2, L2):
            goal_box(sx, self._pad, self._paw / 2.0)   # 페널티 에어리어(깊이 _pad / 폭 _paw)
            goal_box(sx, self._gad, self._gaw / 2.0)   # 골 에어리어(깊이 _gad / 폭 _gaw)

        # 페널티 마크(+자), 양쪽
        for sign in (-1.0, 1.0):
            cx = sign * (L2 - self._pmd)
            h = self._pms / 2.0
            seg(cx - h, 0.0, cx + h, 0.0)
            seg(cx, -h, cx, h)

        arr.markers.append(lines)

        # ---- (2) 센터 서클 ----
        circle = Marker()
        circle.header.frame_id = self._frame
        circle.ns = 'field_lines'; circle.id = 1
        circle.type = Marker.LINE_STRIP
        circle.action = Marker.ADD
        circle.scale.x = self._lt
        circle.color = white
        circle.pose.orientation.w = 1.0
        N = 72
        for i in range(N + 1):
            th = 2.0 * math.pi * i / N
            circle.points.append(_pt(self._ccr * math.cos(th),
                                     self._ccr * math.sin(th)))
        arr.markers.append(circle)

        # ---- (3) 잔디 외곽(5x3) 점선 느낌의 어두운 라인 ----
        if self._draw_turf:
            dark = ColorRGBA(r=0.35, g=0.45, b=0.35, a=0.8)
            turf = self._line_marker('turf', 2, dark)
            tL2, tW2 = self._tL / 2.0, self._tW / 2.0

            def tseg(x1, y1, x2, y2):
                turf.points.append(_pt(x1, y1))
                turf.points.append(_pt(x2, y2))

            tseg(-tL2, -tW2,  tL2, -tW2)
            tseg(-tL2,  tW2,  tL2,  tW2)
            tseg(-tL2, -tW2, -tL2,  tW2)
            tseg( tL2, -tW2,  tL2,  tW2)
            arr.markers.append(turf)

        # ---- (4) PVC 골 기둥 ----
        if self._draw_goals:
            gy = self._gw / 2.0
            colors = {
                -L2: ColorRGBA(r=0.2, g=0.4, b=1.0, a=1.0),   # 파랑 진영
                 L2: ColorRGBA(r=1.0, g=0.85, b=0.1, a=1.0),  # 노랑 진영
            }
            mid = 10
            for gx in (-L2, L2):
                for sy in (-gy, gy):
                    post = Marker()
                    post.header.frame_id = self._frame
                    post.ns = 'goals'; post.id = mid; mid += 1
                    post.type = Marker.CYLINDER
                    post.action = Marker.ADD
                    post.pose.position = _pt(gx, sy, self._gph / 2.0)
                    post.pose.orientation.w = 1.0
                    post.scale.x = self._gpr * 2.0
                    post.scale.y = self._gpr * 2.0
                    post.scale.z = self._gph
                    post.color = colors[gx]
                    arr.markers.append(post)

        return arr


def main(args=None):
    rclpy.init(args=args)
    node = FieldLinesNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # Ctrl+C / 외부 shutdown — 정상 종료 경로
        pass
    finally:
        node.destroy_node()
        # 시그널 핸들러가 이미 shutdown 했다면 다시 호출하지 않는다
        # (rcl_shutdown already called 방지).
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()