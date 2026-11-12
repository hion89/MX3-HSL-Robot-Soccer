#!/usr/bin/env python3
# ==========================================================================
# game_state_to_lock_node.py
# --------------------------------------------------------------------------
# /game_state(GameState) 를 받아서 twist_mux 의 game_stop 락 토픽
# (game_stop_lock, std_msgs/Bool)을 켜고 끈다.
#
# [twist_mux 락 동작 — twist_mux_topics.yaml 참고]
#   game_stop 락은 priority 50. Bool=True 가 되면 그 priority 이하의 입력
#   (behavior=10, navigation=20)을 전부 막아 로봇을 0속도로 정지시킨다.
#   teleop(100)은 막지 않으므로, 정지 중에도 사람이 수동으로 빼낼 수 있다.
#
# [정지(True) 조건]
#   - state ∈ {INITIAL, SET, FINISHED}        (경기 전/세트/종료 → 정지)
#   - game_phase == TIMEOUT                    (작전타임 → 정지)
#   - 내 선수의 penalty != NONE                (반칙 퇴장 중 → 정지)
#   - /game_state 가 stale(= GameController 끊김)  → fail-safe 정지
#   ※ READY 는 포메이션으로 *걸어가야* 하므로 정지하지 않는다(락 OFF).
#   ※ PLAYING 도 락 OFF (자율 주행 허용).
#   ※ stopped/set_play 의 세밀한 처리는 전략 FSM(다음 단계)에서 다룬다.
#     여기서는 "움직여도 되나 / 절대 멈춰야 하나"의 안전 게이트만 담당.
#
# [발행]
#   game_stop_lock (std_msgs/Bool)  — twist_mux 입력. 주기적으로 재발행.
#
# [파라미터]
#   player_number     (int,   기본 1)    내 선수 번호. penalties[player_number-1] 확인.
#   stale_timeout_sec (float, 기본 2.0)  이 시간 이상 새 /game_state 없으면 정지.
#   publish_rate_hz   (float, 기본 10.0) 락 상태 주기 재발행(락 timeout=0 대응).
#
# [검증]
#   ros2 topic echo game_stop_lock
#   → GameController 로 INITIAL/SET 줬을 때 data: true, PLAYING 일 때 false 확인.
#   → 실기에서는 SET 에서 로봇이 얼고 PLAYING 에서 풀리는지 확인.
# ==========================================================================
import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool, Empty
from humanoid_behavior_interfaces.msg import GameState


class GameStateToLock(Node):
    def __init__(self):
        super().__init__('game_state_to_lock')

        # ── 파라미터 ──
        self.player_number = int(self.declare_parameter('player_number', 1).value)
        self.stale_timeout = float(self.declare_parameter('stale_timeout_sec', 2.0).value)
        rate = float(self.declare_parameter('publish_rate_hz', 10.0).value)
        # 토크온(시작 버튼)으로 재진입 대기 hold 를 걸 때 구독할 이벤트 토픽.
        # button_input_node 가 시작 버튼(=토크온/robot_init) 시 여기에 1회 펄스를 쏜다.
        self.start_event_topic = str(
            self.declare_parameter('start_event_topic', '/buttons/start_event').value)

        # ── 통신 ──
        self.pub = self.create_publisher(Bool, 'game_stop_lock', 10)
        self.create_subscription(GameState, '/game_state', self._cb_state, 10)
        # 재진입 대기 트리거: 토크온(시작 버튼) 펄스.
        self.create_subscription(
            Empty, self.start_event_topic, self._cb_start_event, 10)

        # ── 내부 상태 ──
        self.last_msg = None
        self.last_recv = None          # 마지막 수신 시각(rclpy Time)
        self.prev_lock = None          # 직전 락 상태(전환 로그용). None=아직 없음
        # 재진입 대기 hold:
        #   퇴장 후 재진입 시 초기위치 보정을 위해 토크온(시작 버튼)을 누르는데,
        #   토크온으로 로봇이 서면 fall_manager 가 NORMAL 로 풀려 strategy 가
        #   곧바로 목적지를 향해 걸어가 버린다("이전 목적지로 돌진"). 이를 막기 위해
        #   토크온을 "재진입 진행" 신호로 보고 game_stop 락을 강제로 유지한다.
        #   해제는 GC 가 penalty 를 풀어 복귀를 허가할 때(= _i_am_penalised False).
        self._reentry_hold = False

        # 주기 발행 타이머: 락 토픽 timeout 이 0(상태 변할 때만)이라 해도,
        # 늦게 뜬 twist_mux 가 현재 값을 받도록 꾸준히 재발행한다.
        self.create_timer(1.0 / max(rate, 1.0), self._tick)

        self.get_logger().info(
            f"game_stop_lock 매핑 시작 — player_number={self.player_number}, "
            f"stale_timeout={self.stale_timeout}s")

    # ----------------------------------------------------------------------
    def _cb_start_event(self, _msg: Empty):
        """토크온(시작 버튼) 펄스 → 재진입 대기 hold 를 건다.

        - PLAYING 중일 때만 건다. 경기 시작 셋업(INITIAL/SET)/종료(FINISHED)에선
          state 락이 이미 정지시키므로 hold 가 불필요하고, 오히려 PLAYING 진입 후
          까지 남으면 정상 시작을 방해할 수 있어 걸지 않는다.
        - 해제는 _compute_lock 에서 GC penalty 해제(복귀 허가) 시 처리한다.
        """
        if self.last_msg is not None and \
                self.last_msg.state == GameState.STATE_PLAYING:
            if not self._reentry_hold:
                self.get_logger().warn(
                    "토크온 감지 → 재진입 대기(hold ON). GC penalty 해제 시 복귀")
            self._reentry_hold = True

    # ----------------------------------------------------------------------
    def _cb_state(self, msg: GameState):
        self.last_msg = msg
        self.last_recv = self.get_clock().now()

    # ----------------------------------------------------------------------
    def _tick(self):
        """현재 경기 상태로부터 락 여부를 계산해 발행한다."""
        lock, reason = self._compute_lock()

        out = Bool()
        out.data = lock
        self.pub.publish(out)

        # 상태가 바뀔 때만 로그(스팸 방지). 첫 발행도 prev_lock=None 이라 찍힌다.
        if lock != self.prev_lock:
            self.get_logger().warn(
                f"game_stop_lock {'ON(정지)' if lock else 'OFF(주행 허용)'} — {reason}")
            self.prev_lock = lock

    # ----------------------------------------------------------------------
    def _compute_lock(self):
        """(lock: bool, reason: str) 반환. 기본값은 안전을 위해 정지(True)."""
        # 아직 패킷 못 받음 → 정지
        if self.last_msg is None or self.last_recv is None:
            return True, 'no_data(아직 수신 없음)'

        # GameController 끊김(stale) → fail-safe 정지
        age = (self.get_clock().now() - self.last_recv).nanoseconds * 1e-9
        if age > self.stale_timeout:
            return True, f'stale({age:.1f}s, GC 끊김 추정)'

        m = self.last_msg

        # ── 재진입 대기 hold 갱신 ──────────────────────────────────
        # penalty 가 풀리면(NONE) GC 가 복귀를 허가한 것 → hold 해제(복귀).
        # PLAYING 이 아니게 되면(하프타임/종료 등) 안전하게 리셋.
        penalised = self._i_am_penalised(m)
        if self._reentry_hold:
            if (not penalised) or (m.state != GameState.STATE_PLAYING):
                self.get_logger().warn(
                    "GC penalty 해제/상태 전환 → 재진입 복귀(hold OFF)")
                self._reentry_hold = False

        # 경기 전/세트/종료 상태 → 정지
        if m.state in (GameState.STATE_INITIAL,
                       GameState.STATE_SET,
                       GameState.STATE_FINISHED):
            return True, f'state={m.state}'

        # 작전타임 → 정지
        if m.game_phase == GameState.GAME_PHASE_TIMEOUT:
            return True, 'phase=TIMEOUT'

        # 내가 패널티 중 → 정지
        if penalised:
            return True, 'penalised(내 선수 패널티)'

        # 재진입 대기(토크온 후 GC 복귀 전) → 정지
        if self._reentry_hold:
            return True, 'reentry_hold(토크온 재진입 대기, GC 복귀 전)'

        # 그 외(READY, PLAYING) → 주행 허용
        return False, f'state={m.state}'

    # ----------------------------------------------------------------------
    def _i_am_penalised(self, m: GameState) -> bool:
        """내 선수 번호의 penalty 가 NONE 이 아니면 True."""
        if not m.team_found:
            return False   # 우리 팀을 못 찾았으면 패널티 판단 보류
        idx = self.player_number - 1   # 선수 번호는 1부터, 배열은 0부터
        if 0 <= idx < len(m.penalties):
            return m.penalties[idx] != GameState.PENALTY_NONE
        return False


def main(args=None):
    rclpy.init(args=args)
    node = GameStateToLock()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
