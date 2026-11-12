#!/usr/bin/env python3
"""
neck_scan_node.py
=================

목(yaw)을 좌우 끝점 사이로 핑퐁(ping-pong) 시키는 탐색 노드.

설계:
  - 사인파처럼 매 tick 위치를 계산해서 흘려보내는 게 아니라,
    "왼쪽 끝 → (도착 대기) → 오른쪽 끝 → (도착 대기) → ..." 식으로
    끝점만 명령한다. AX-18A 의 내장 MOVING_SPEED 가 알아서 속도를
    제한하면서 두 점 사이를 부드럽게 보간하므로, 외부에서 사인파를
    흉내낼 필요가 없다. 토픽 트래픽도, MCU 큐 부담도 최소화.

  - 도착 판정: AX-18A 의 Present Position 을 이 노드가 직접 읽지
    못하므로(피드백 채널이 ROS 단까지 노출돼 있지 않음) "충분히
    오래 기다린다"는 시간 기반 정책을 쓴다.
       sweep_period = (왼쪽끝↔오른쪽끝 1회 이동에 걸리는 시간)
    이걸 파라미터로 노출해 두면 실제 모터 속도에 맞춰 튜닝 가능.

대략적 시간 계산 (참고용, 실제값은 튜닝 필요):
  - AX-18A MOVING_SPEED 단위 ≈ 0.111 rpm/unit
  - 10 rpm ≈ 60 deg/s
  - 풀스윙 180° (±90°) → 180/60 = 3 s
  - 가/감속 + 안전마진 → 기본값 3.5 s 권장

토픽:
  구독:
    /neck/scan_enable    (Bool)
        true  : 스캔 시작 (이미 켜져 있으면 무시)
        false : 스캔 중지. 현재 위치 그대로 둠 (정면 복귀 X).
                정면 복귀가 필요하면 robot_init 서비스를 따로 부르면 됨.
  발행:
    /neck/yaw_deg        (Float32)  스캔 endpoint 명령
    /neck/moving_speed   (UInt16)   스캔 시작 시 1회 설정 (raw, AX 단위)
    /neck/torque_enable  (Bool)     스캔 시작 시 1회 토크 ON

파라미터 (ros2 param):
  yaw_min_deg      (float, def -75.0)  왼쪽 끝 (REP-103: + = 왼쪽)
                                       neck_control_node 의 한계(±90°)
                                       안쪽으로 잡는 게 좋음. 기구
                                       hard-stop 충돌 방지.
  yaw_max_deg      (float, def  75.0)  오른쪽 끝 — 부호 주의:
                                       오른쪽이 음수이므로 yaw_min < yaw_max
                                       가 되려면 변수명을 'min'/'max' 가
                                       아니라 'left'/'right' 로 해석해야 함.
                                       여기서는 통상적으로 min<max 로 두고
                                       내부에서 left/right 으로 매핑.
  sweep_period_sec (float, def 3.5)   한쪽 끝→반대쪽 끝 1회 이동 추정 시간.
                                       이 주기마다 목적지를 토글한다.
                                       실제 모터 속도보다 짧으면 이전
                                       명령이 완료되기 전에 새 명령이
                                       들어와 모터가 끊기듯 움직임.
                                       → 모자라면 늘려라.
  moving_speed_raw (int, def 90)      스캔 시 적용할 AX MOVING_SPEED
                                       (0.111 rpm/unit). 90 ≈ 10 rpm.
                                       0 은 "최대속도(무제한)" 이라 금지.
                                       1~1023 범위로 강제.

상태 머신:
  IDLE   : 정지 상태. enable=true 받으면 INIT → SWEEPING.
  INIT   : 시작 1회 셋업 (torque ON, speed 설정, 첫 목적지 = LEFT 발행).
  SWEEPING : 매 sweep_period_sec 마다 목적지를 LEFT ↔ RIGHT 토글.
             enable=false 받으면 IDLE 로.
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, UInt16, Bool


# 상태 머신 식별자 (문자열 그대로 로깅에 쓰기 좋게 상수로)
STATE_IDLE     = "IDLE"
STATE_SWEEPING = "SWEEPING"


def _clamp(v, lo, hi):
    """일반 clamp. ROS 메시지 값이나 파라미터 값 안전 처리에 사용."""
    return max(lo, min(hi, v))


class NeckScanNode(Node):
    def __init__(self):
        super().__init__('neck_scan_node')

        # ── 파라미터 선언 ────────────────────────────────────
        # ros2 run 시 --ros-args -p yaw_min_deg:=-60.0 식으로 덮어쓰기 가능.
        # 또는 ros2 param set /neck_scan_node sweep_period_sec 4.0 등으로
        # 런타임에 바꿀 수도 있다 (단, 본 노드는 변경 알림을 구독하지
        # 않으므로 변경 후 enable=false → true 로 재시작해야 반영).
        self.declare_parameter('yaw_min_deg',      -75.0)  # 오른쪽 끝 (음수)
        self.declare_parameter('yaw_max_deg',       75.0)  # 왼쪽 끝 (양수)
        self.declare_parameter('sweep_period_sec',   3.5)  # 한쪽→반대쪽 이동 시간
        self.declare_parameter('moving_speed_raw',    90)  # AX raw, ~10 rpm

        # 파라미터 캐싱 (생성 시점 1회). 재시작 없이 반영하고 싶으면
        # add_on_set_parameters_callback 으로 핸들러 추가 필요.
        self._yaw_min  = float(self.get_parameter('yaw_min_deg').value)
        self._yaw_max  = float(self.get_parameter('yaw_max_deg').value)
        self._period   = float(self.get_parameter('sweep_period_sec').value)
        self._spd_raw  = int(self.get_parameter('moving_speed_raw').value)

        # 안전 클램프
        #   - sweep_period 가 너무 작으면 명령이 끊겨서 모터가 덜덜거림.
        #     0.2초 미만은 사실상 의미 없음. 최소 0.2초로.
        #   - moving_speed_raw=0 은 AX-18A 에서 "최대속도(무제한)" 라
        #     반드시 1 이상으로. 1023 초과도 잘라낸다.
        if self._period < 0.2:
            self.get_logger().warn(
                f"sweep_period_sec={self._period} 너무 짧음. 0.2 로 클램프."
            )
            self._period = 0.2
        self._spd_raw = _clamp(int(self._spd_raw), 1, 1023)

        # yaw_min < yaw_max 보장 (사용자가 거꾸로 입력해도 동작하도록)
        if self._yaw_min > self._yaw_max:
            self.get_logger().warn(
                f"yaw_min_deg({self._yaw_min}) > yaw_max_deg({self._yaw_max}) — 자동 swap"
            )
            self._yaw_min, self._yaw_max = self._yaw_max, self._yaw_min

        # ── 퍼블리셔 ───────────────────────────────────────
        # neck_control_node 가 구독하는 토픽들. 동일 노드에서 init/zero
        # 서비스(body_motor_controller_node)와 함께 발행해도 마지막 publish
        # 가 이기므로 충돌 없음.
        self.pub_yaw    = self.create_publisher(Float32, '/neck/yaw_deg',        10)
        self.pub_speed  = self.create_publisher(UInt16,  '/neck/moving_speed',   10)
        self.pub_torque = self.create_publisher(Bool,    '/neck/torque_enable',  10)

        # ── 구독자 ─────────────────────────────────────────
        self.create_subscription(Bool, '/neck/scan_enable', self._on_enable, 10)

        # ── 내부 상태 ──────────────────────────────────────
        self._state = STATE_IDLE
        # 다음에 보낼 목적지가 LEFT 인지 RIGHT 인지.
        # 시작 직후 첫 발행을 LEFT 로 하기 위해 True 로 초기화.
        # (어느 쪽이든 상관없지만, 일관성 위해 고정.)
        self._next_target_is_left = True
        # 타이머 핸들. enable=true 일 때만 살아 있고 false 가 되면 destroy.
        # 매 sweep_period_sec 마다 _on_tick 호출.
        self._timer = None

        self.get_logger().info(
            "neck_scan_node ready. "
            f"range=[{self._yaw_min:+.1f}°, {self._yaw_max:+.1f}°], "
            f"period={self._period:.2f}s, "
            f"speed_raw={self._spd_raw} (~{self._spd_raw*0.111:.1f} rpm). "
            "Publish true to /neck/scan_enable to start."
        )

    # ════════════════════════════════════════════════════════════════
    # enable 토픽 콜백
    # ════════════════════════════════════════════════════════════════
    def _on_enable(self, msg: Bool):
        """
        /neck/scan_enable 콜백.
          - true 가 들어오면: IDLE 이면 SWEEPING 으로 전이 + 초기 setup.
                              이미 SWEEPING 이면 무시 (중복 시작 방지).
          - false 가 들어오면: SWEEPING 이면 타이머 정지 → IDLE.
                              이미 IDLE 이면 무시.
        """
        want = bool(msg.data)
        if want:
            if self._state == STATE_SWEEPING:
                # 이미 돌고 있는데 또 true 가 들어옴 — 무시.
                # (재시작이 필요하면 false → true 순서로 보낼 것)
                self.get_logger().debug("enable=true received but already SWEEPING — ignored")
                return
            self._start_sweep()
        else:
            if self._state == STATE_IDLE:
                self.get_logger().debug("enable=false received but already IDLE — ignored")
                return
            self._stop_sweep()

    # ════════════════════════════════════════════════════════════════
    # 시작/정지
    # ════════════════════════════════════════════════════════════════
    def _start_sweep(self):
        """
        SWEEPING 진입 시퀀스.
          1) 토크 ON  — 안 켜져 있으면 그 뒤 명령들이 다 무시됨.
          2) MOVING_SPEED 설정 — GoalPosition 보다 반드시 먼저.
                                  (순서 뒤집으면 풀스피드로 출발하고
                                   나서야 속도가 적용됨)
          3) 첫 목적지 발행 (LEFT = yaw_max, REP-103: + = 왼쪽).
          4) sweep_period_sec 주기 타이머 생성.

        모든 발행은 fire-and-forget — neck_control_node 가 받아서 MCU 로
        보낼 때까지 수십 ms 소요되지만, 어차피 첫 sweep 도착 추정 시간이
        수 초 단위라 무시 가능.
        """
        self.get_logger().info(
            f"scan START: first goal LEFT={self._yaw_max:+.1f}°"
        )

        # 1) torque ON (멱등 — 이미 켜져 있으면 부작용 없음)
        torq = Bool()
        torq.data = True
        self.pub_torque.publish(torq)

        # 2) MOVING_SPEED — GoalPosition 보다 먼저!
        spd = UInt16()
        spd.data = self._spd_raw
        self.pub_speed.publish(spd)

        # 3) 첫 endpoint = LEFT (yaw_max, 양수)
        #    그 다음 tick 에서 RIGHT 로 토글되도록 플래그 미리 뒤집어 둠.
        self._publish_yaw(self._yaw_max)
        self._next_target_is_left = False  # 다음 번엔 RIGHT

        # 4) 주기 타이머 시작
        self._timer = self.create_timer(self._period, self._on_tick)

        self._state = STATE_SWEEPING

    def _stop_sweep(self):
        """
        IDLE 복귀.
          - 타이머만 정지. 현재 위치는 그대로 둔다 (정면 복귀 X).
            만약 정지 시 정면으로 돌리고 싶다면 여기서 yaw_deg=0.0 을
            1회 발행하면 되지만, "스캔 중단 = 마지막 위치 유지" 가 다른
            FSM 동작과 더 합 좋음 (예: 공을 본 직후 멈췄을 때 그 방향
            그대로 유지하고 추적 단계로 넘어감).
          - 토크는 끄지 않는다. 목이 풀려서 머리가 꺾이는 걸 방지.
            토크를 끄려면 motor-off (cb_torque_enable) 를 따로 호출.
        """
        self.get_logger().info("scan STOP (holding current pose)")
        if self._timer is not None:
            self._timer.cancel()
            self.destroy_timer(self._timer)
            self._timer = None
        self._state = STATE_IDLE

    # ════════════════════════════════════════════════════════════════
    # 주기 tick: endpoint 토글
    # ════════════════════════════════════════════════════════════════
    def _on_tick(self):
        """
        sweep_period_sec 마다 호출. LEFT ↔ RIGHT 를 토글하면서 발행.

        주의:
          - 첫 호출은 start 직후 sweep_period_sec 가 지나서야 들어온다.
            즉, _start_sweep() 에서 이미 LEFT 를 한 번 보냈으므로 첫
            tick 의 일은 RIGHT 발행. _next_target_is_left 플래그가
            그 의도를 보존한다.
          - publish 후 토글하는 순서가 맞다. 토글 후 publish 하면 의미가
            "다음에 보낼 위치" 가 되어버려 헷갈림. (둘 다 동작은 같지만
            상태 변수의 의미를 명확히 하기 위해 "지금 보낼 방향" 으로 해석.)
        """
        if self._next_target_is_left:
            target = self._yaw_max     # + = 왼쪽
            label  = "LEFT"
        else:
            target = self._yaw_min     # - = 오른쪽
            label  = "RIGHT"

        self._publish_yaw(target)
        self.get_logger().debug(f"tick → {label}={target:+.1f}°")

        # 다음 tick 을 위해 방향 반전
        self._next_target_is_left = not self._next_target_is_left

    # ════════════════════════════════════════════════════════════════
    # 발행 헬퍼
    # ════════════════════════════════════════════════════════════════
    def _publish_yaw(self, deg: float):
        """
        /neck/yaw_deg 1회 발행. neck_control_node 내부에서 한 번 더 clamp
        하지만, 여기서도 안전상 한 번 더 잘라낸다 (파라미터 사용자가
        실수로 ±90° 밖을 줬을 때 방어).
        """
        msg = Float32()
        # neck_control_node 의 YAW_DEG_MIN/MAX = ±90° 와 동일한 범위.
        # 파라미터가 그 안쪽이면 영향 없고, 밖이면 잘라낸다.
        msg.data = _clamp(float(deg), -90.0, 90.0)
        self.pub_yaw.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = NeckScanNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Ctrl+C 는 정상 종료. 스택트레이스 안 찍음.
        pass
    finally:
        # 종료 직전 타이머가 살아 있으면 정리 (잡힌 자원 누수 방지)
        if node._timer is not None:
            node._timer.cancel()
            node.destroy_timer(node._timer)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()