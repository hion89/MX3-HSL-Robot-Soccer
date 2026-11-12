#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  nav_cmd_bridge_node.py
#
#  Nav2(컨트롤러)가 내는 cmd_vel 을 "보행 정책 명령값"으로 변환하는 브릿지.
#
#  [왜 필요한가]
#  --------------------------------------------------------------------------
#  - Nav2 와 /odom 은 모두 m/s(실제 속도) 단위로 동작한다.
#  - 그러나 actor_onnx_runner 는 cmd_vel 을 "정책 명령값"으로 받는다.
#    (명령 0.3 → 실제 0.2353 m/s 처럼 명령값과 실제속도가 다르다)
#  - 따라서 Nav2 가 원하는 "실제 속도(m/s)"를 정책이 이해하는 "명령값"으로
#    역변환(inverse calibration)해 줘야 로봇이 의도한 속도로 움직인다.
#  - 또한 보행에는 데드존(작은 명령은 안 움직임)과 전/후진 비대칭이 있어,
#    Nav2 가 데드존 아래 속도를 요청하면 멈춰버린다. 이를 여기서 보정한다.
#
#  [흐름]
#     Nav2 controller (/cmd_vel → remap → /cmd_vel_nav_raw, m/s)
#        → [이 노드: 역calib + 데드존 보정]
#        → /cmd_vel_nav (정책 명령값) → twist_mux(navigation 입력) → ... → actor
#
#  [역변환 원리]
#     전진 측정점:  실제 [0.1857, 0.2353, 0.2857]  ↔  명령 [0.2, 0.3, 0.4]
#     → 원하는 실제속도를 이 점들로 역보간해 명령값을 얻는다.
#     → 데드존(최소 작동 실제속도 미만)인데 "움직여야" 하면 최소 작동 명령으로 올린다.
#     → 거의 0(stop_eps 미만) 요청이면 정지(0).
#
#  [회전(wz)]
#     실제 wz ≈ wz_scale * 명령 wz  (측정: 명령 1.0 → 실제 0.698 rad/s, scale≈0.7)
#     → 명령 = 원하는 실제 wz / wz_scale. 정책 학습 범위로 clip.
# =============================================================================

import math

import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class NavCmdBridgeNode(Node):
    def __init__(self):
        super().__init__('nav_cmd_bridge_node')

        # ---------------------------------------------------------------------
        #  파라미터 (odom_calib.yaml 의 nav_cmd_bridge_node 섹션에서 주입)
        # ---------------------------------------------------------------------
        # 전진 역calib 측정점: 실제속도(m/s) ↔ 명령값. real 은 오름차순.
        self.declare_parameter('fwd_real', [0.1857, 0.2353, 0.2857])
        self.declare_parameter('fwd_cmd',  [0.2, 0.3, 0.4])
        # 데드존/정지 처리
        self.declare_parameter('min_real_fwd', 0.1857)  # 이 미만 실제속도는 데드존
        self.declare_parameter('min_cmd_fwd', 0.2)      # 데드존인데 움직여야 하면 줄 최소 명령
        self.declare_parameter('stop_eps', 0.03)        # 이보다 작은 |vx| 요청은 정지로 간주
        # 후진: 데이터가 부족하고 RPP 에서 보통 비활성. 들어오면 보수적으로 최소 명령만.
        self.declare_parameter('allow_reverse', False)
        self.declare_parameter('min_cmd_bwd', 0.3)      # 후진 최소 작동 명령(관측상 0.3부터 움직임)
        # 회전
        self.declare_parameter('wz_scale', 0.7)         # 실제 wz = wz_scale * 명령
        self.declare_parameter('max_cmd_wz', 1.5)       # 명령 wz 상한(정책 학습 범위)
        self.declare_parameter('wz_eps', 0.02)          # 이보다 작은 |wz| 요청은 0

        # 제자리 회전 데드존 보상:
        #   actor 는 전진 없이 회전만 할 때 |명령wz| 가 일정값 미만이면 회전을 못 한다(관측).
        #   전진이 거의 없고(제자리) 회전 요청이 있으면 회전 명령을 최소 작동값으로 올린다.
        self.declare_parameter('min_cmd_wz_spin', 1.2)  # 제자리 회전 최소 작동 명령
        self.declare_parameter('spin_vx_eps', 0.05)     # |vx명령| 이 미만이면 제자리 회전으로 간주
        self.declare_parameter('min_cmd_wz_curve', 0.1) # 곡선 주행(전진+회전) 시 회전 최소 명령

        g = self.get_parameter
        self._fwd_real = np.array(g('fwd_real').value, dtype=float)
        self._fwd_cmd  = np.array(g('fwd_cmd').value, dtype=float)
        self._min_real_fwd = float(g('min_real_fwd').value)
        self._min_cmd_fwd = float(g('min_cmd_fwd').value)
        self._stop_eps = float(g('stop_eps').value)
        self._allow_reverse = bool(g('allow_reverse').value)
        self._min_cmd_bwd = float(g('min_cmd_bwd').value)
        self._wz_scale = float(g('wz_scale').value)
        self._max_cmd_wz = float(g('max_cmd_wz').value)
        self._wz_eps = float(g('wz_eps').value)
        self._min_cmd_wz_spin = float(g('min_cmd_wz_spin').value)
        self._spin_vx_eps = float(g('spin_vx_eps').value)
        self._min_cmd_wz_curve = float(g('min_cmd_wz_curve').value)

        if not np.all(np.diff(self._fwd_real) > 0):
            self.get_logger().error('fwd_real 이 오름차순이 아닙니다 — 역보간 부정확.')

        # 입력: Nav2 컨트롤러 출력(m/s) / 출력: 정책 명령값
        self.create_subscription(Twist, '/cmd_vel_nav_raw', self._cb, 10)
        self._pub = self.create_publisher(Twist, '/cmd_vel_nav', 10)

        self.get_logger().info(
            'nav_cmd_bridge 시작: /cmd_vel_nav_raw(m/s) -> 역calib -> /cmd_vel_nav(명령값)')

    # =========================================================================
    def _vx_ms_to_cmd(self, vx_ms: float) -> float:
        """원하는 실제 전진속도(m/s)를 정책 명령값으로 역변환."""
        if abs(vx_ms) < self._stop_eps:
            return 0.0
        if vx_ms > 0.0:
            if vx_ms < self._min_real_fwd:
                # 데드존이지만 전진해야 함 → 최소 작동 명령으로 올림
                return self._min_cmd_fwd
            # 측정점으로 역보간(범위 밖은 양끝 명령으로 clamp)
            return float(np.interp(vx_ms, self._fwd_real, self._fwd_cmd))
        else:
            # 후진 요청
            if not self._allow_reverse:
                return 0.0
            return -self._min_cmd_bwd

    def _wz_to_cmd(self, wz_ms: float) -> float:
        """원하는 실제 회전속도(rad/s)를 명령값으로 역변환."""
        if abs(wz_ms) < self._wz_eps:
            return 0.0
        cmd = wz_ms / self._wz_scale
        return float(np.clip(cmd, -self._max_cmd_wz, self._max_cmd_wz))

    def _cb(self, msg: Twist):
        out = Twist()
        vx_cmd = self._vx_ms_to_cmd(float(msg.linear.x))
        wz_cmd = self._wz_to_cmd(float(msg.angular.z))

        # 회전 명령 데드존 보상(전진 여부로 하한을 달리한다):
        #   - 제자리 회전(|vx명령| < spin_vx_eps): actor 가 큰 명령(min_cmd_wz_spin)이라야 돈다.
        #   - 곡선 주행(전진 있음): 작은 회전도 동작하지만, 너무 작은 명령(< min_cmd_wz_curve)은
        #     효과가 없으므로 최소 min_cmd_wz_curve 로 올려 회전이 확실히 나가게 한다.
        #   둘 다 부호는 유지하고, 회전 요청이 있을 때(wz_cmd != 0)만 적용한다.
        if abs(wz_cmd) > 0.0:
            floor = self._min_cmd_wz_spin if abs(vx_cmd) < self._spin_vx_eps \
                    else self._min_cmd_wz_curve
            if abs(wz_cmd) < floor:
                wz_cmd = math.copysign(floor, wz_cmd)

        out.linear.x = vx_cmd
        out.linear.y = 0.0   # vy 는 오도메트리/주행 모두 미사용
        out.angular.z = wz_cmd
        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = NavCmdBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()