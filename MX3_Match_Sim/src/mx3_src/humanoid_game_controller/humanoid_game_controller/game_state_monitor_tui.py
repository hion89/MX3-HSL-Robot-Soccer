#!/usr/bin/env python3
"""
game_state_monitor_tui.py

KHUBER GameController 상태 TUI 모니터
/game_state (humanoid_behavior_interfaces/GameState) 구독 → 터미널 대시보드 표시
game_stop_lock (std_msgs/Bool) 도 함께 구독해 정지/주행 상태를 한눈에 표시.

실행:
    ros2 run humanoid_game_controller game_state_monitor_tui
    # 내 로봇 강조 표시를 원하면:
    ros2 run humanoid_game_controller game_state_monitor_tui --ros-args -p player_number:=4

motor_monitor_tui 와 동일한 curses 스타일로, enum 코드를 사람이 읽는 이름으로
번역해서 보여준다(예: state 3 → PLAYING, penalty 3 → MOTION_IN_STOP).

종료: Q
"""

import time
import curses
import threading
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from humanoid_behavior_interfaces.msg import GameState


# ── enum 코드 → 사람이 읽는 이름 (RoboCupGameControlData.h 와 1:1) ──
STATE_NAMES = {0: 'INITIAL', 1: 'READY', 2: 'SET', 3: 'PLAYING', 4: 'FINISHED'}
PHASE_NAMES = {0: 'NORMAL', 1: 'PENALTY_SHOOTOUT', 2: 'EXTRA_TIME', 3: 'TIMEOUT'}
SETPLAY_NAMES = {
    0: 'NONE', 1: 'DIRECT_FREE_KICK', 2: 'INDIRECT_FREE_KICK', 3: 'PENALTY_KICK',
    4: 'THROW_IN', 5: 'GOAL_KICK', 6: 'CORNER_KICK',
}
COMPETITION_NAMES = {0: 'SMALL', 1: 'MIDDLE', 2: 'LARGE'}
PENALTY_NAMES = {
    0: 'NONE', 1: 'ILLEGAL_POSITIONING', 2: 'MOTION_IN_SET', 3: 'MOTION_IN_STOP',
    4: 'LOCAL_GAME_STUCK', 5: 'INCAPABLE_ROBOT', 6: 'PICK_UP', 7: 'BALL_HOLDING',
    8: 'LEAVING_THE_FIELD', 9: 'PLAYING_WITH_ARMS_HANDS', 10: 'PUSHING',
    11: 'CAUTIONED', 12: 'SENT_OFF', 13: 'SUBSTITUTE',
}


def fmt_time(secs: int) -> str:
    """초 → mm:ss (음수면 그대로 표시)."""
    if secs < 0:
        return f"-{abs(secs) // 60:02d}:{abs(secs) % 60:02d}"
    return f"{secs // 60:02d}:{secs % 60:02d}"


class GameStateMonitorNode(Node):
    """/game_state 와 game_stop_lock 을 구독해 최신값만 보관한다(렌더는 별도 루프)."""

    def __init__(self):
        super().__init__('game_state_monitor_tui')

        # 내 로봇 강조용(선택). 0이면 강조 안 함.
        self.player_number = int(self.declare_parameter('player_number', 0).value)

        self.lock = threading.Lock()
        self.latest: Optional[GameState] = None    # 최신 GameState
        self.last_recv = 0.0                        # 최신 GameState 수신 시각
        self.latest_lock: Optional[bool] = None     # 최신 game_stop_lock 값
        self.last_lock_recv = 0.0

        self.create_subscription(GameState, '/game_state', self._cb_state, 10)
        self.create_subscription(Bool, 'game_stop_lock', self._cb_lock, 10)

    def _cb_state(self, msg: GameState):
        with self.lock:
            self.latest = msg
            self.last_recv = time.time()

    def _cb_lock(self, msg: Bool):
        with self.lock:
            self.latest_lock = msg.data
            self.last_lock_recv = time.time()


def draw_dashboard(stdscr, node: GameStateMonitorNode):
    """curses 메인 루프 — 200ms 주기로 화면을 다시 그린다."""
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(200)

    # ── 색상 초기화 (motor_monitor_tui 와 동일 팔레트) ──
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)    # 정상/주행
    curses.init_pair(2, curses.COLOR_YELLOW, -1)   # 주의(SET/READY 등)
    curses.init_pair(3, curses.COLOR_RED, -1)       # 정지/패널티/위험
    curses.init_pair(4, curses.COLOR_CYAN, -1)      # 헤더
    curses.init_pair(5, curses.COLOR_WHITE, -1)     # 일반
    curses.init_pair(6, curses.COLOR_MAGENTA, -1)   # 강조(내 로봇/킥 주도)

    CLR_OK   = curses.color_pair(1)
    CLR_WARN = curses.color_pair(2)
    CLR_CRIT = curses.color_pair(3)
    CLR_HEAD = curses.color_pair(4) | curses.A_BOLD
    CLR_NORM = curses.color_pair(5)
    CLR_DIM  = curses.color_pair(5) | curses.A_DIM
    CLR_ACC  = curses.color_pair(6) | curses.A_BOLD

    def state_color(state: int):
        """경기 상태별 색: PLAYING=초록, READY=노랑, SET/INITIAL/FINISHED=빨강."""
        if state == GameState.STATE_PLAYING:
            return CLR_OK
        if state == GameState.STATE_READY:
            return CLR_WARN
        return CLR_CRIT

    def section(stdscr, row, w, title):
        """'── TITLE ──────' 형태의 섹션 헤더 한 줄. 다음 row 반환."""
        line = f"── {title} " + "─" * max(0, (min(w, 64) - len(title) - 5))
        try:
            stdscr.addstr(row, 2, line[:w - 3], CLR_DIM)
        except curses.error:
            pass
        return row + 1

    def put(stdscr, row, col, text, attr=CLR_NORM):
        """안전 addstr (터미널이 작아도 죽지 않게)."""
        try:
            stdscr.addstr(row, col, text, attr)
        except curses.error:
            pass

    while True:
        rclpy.spin_once(node, timeout_sec=0.0)

        key = stdscr.getch()
        if key in (ord('q'), ord('Q')):
            break

        stdscr.erase()
        h, w = stdscr.getmaxyx()

        # ── 타이틀 ──
        title = " KHUBER GameController Monitor "
        put(stdscr, 0, max(0, (w - len(title)) // 2), title, CLR_HEAD)

        with node.lock:
            m = node.latest
            last_t = node.last_recv
            lock_val = node.latest_lock
            last_lock_t = node.last_lock_recv

        # ── 연결 상태 (좌측) + 패킷번호 (우측) ──
        age = time.time() - last_t if last_t > 0 else -1.0
        if age < 0:
            put(stdscr, 1, 2, "  Waiting for /game_state ...  ", CLR_WARN)
            stdscr.refresh()
            continue
        elif age > 2.0:
            put(stdscr, 1, 2, f"  STALE — no packet for {age:.0f}s (GC 끊김?)  ", CLR_CRIT)
        else:
            put(stdscr, 1, 2, f"  ● Live ({age:.1f}s ago)  ", CLR_OK)

        if m is not None:
            put(stdscr, 1, max(2, w - 16), f"Packet #{m.packet_number}", CLR_DIM)

        if m is None:
            stdscr.refresh()
            continue

        row = 3

        # ══ GAME ══
        row = section(stdscr, row, w, "GAME")
        put(stdscr, row, 4, "State    : ", CLR_DIM)
        put(stdscr, row, 15, f"{STATE_NAMES.get(m.state, '?'):<10s} ({m.state})", state_color(m.state))
        row += 1
        put(stdscr, row, 4, "Phase    : ", CLR_DIM)
        phase_clr = CLR_CRIT if m.game_phase == GameState.GAME_PHASE_TIMEOUT else CLR_NORM
        put(stdscr, row, 15, f"{PHASE_NAMES.get(m.game_phase, '?'):<16s} ({m.game_phase})", phase_clr)
        row += 1
        put(stdscr, row, 4, "Set Play : ", CLR_DIM)
        sp_clr = CLR_NORM if m.set_play == 0 else CLR_WARN
        put(stdscr, row, 15, f"{SETPLAY_NAMES.get(m.set_play, '?'):<18s} ({m.set_play})", sp_clr)
        row += 1
        put(stdscr, row, 4, "Stopped  : ", CLR_DIM)
        put(stdscr, row, 15, "YES" if m.stopped else "NO", CLR_WARN if m.stopped else CLR_OK)
        half = "1st" if m.first_half else "2nd"
        put(stdscr, row, 28, f"Half {half}   Time {fmt_time(m.secs_remaining)}   2nd {m.secondary_time}s", CLR_DIM)
        row += 2

        # ══ TEAMS ══
        row = section(stdscr, row, w, "TEAMS")
        put(stdscr, row, 4, f"Score    : US({m.our_team_number}) ", CLR_DIM)
        put(stdscr, row, 18, f"{m.our_score}", CLR_ACC)
        put(stdscr, row, 20, " : ", CLR_DIM)
        put(stdscr, row, 23, f"{m.opponent_score}", CLR_NORM)
        put(stdscr, row, 26, " OPP", CLR_DIM)
        row += 1
        put(stdscr, row, 4, "Kicking  : ", CLR_DIM)
        put(stdscr, row, 15, "US" if m.we_are_kicking else "THEM",
            CLR_ACC if m.we_are_kicking else CLR_NORM)
        put(stdscr, row, 24, f"Found: {'YES' if m.team_found else 'NO'}",
            CLR_OK if m.team_found else CLR_CRIT)
        put(stdscr, row, 38, f"GK: #{m.our_goalkeeper}", CLR_DIM)
        put(stdscr, row, 48, f"{COMPETITION_NAMES.get(m.competition_type, '?')}  "
                             f"{m.players_per_team}p", CLR_DIM)
        row += 2

        # ══ OUR ROBOTS ══
        row = section(stdscr, row, w, "OUR ROBOTS")
        if not m.team_found:
            put(stdscr, row, 4, "(team_number 가 패킷과 안 맞아 선수 정보 없음)", CLR_CRIT)
            row += 1
        else:
            for i, pen in enumerate(m.penalties):
                if row >= h - 3:
                    break
                player_no = i + 1
                is_gk = (player_no == m.our_goalkeeper)
                is_me = (node.player_number != 0 and player_no == node.player_number)

                tag = " GK " if is_gk else "    "
                put(stdscr, row, 4, f"#{player_no}{tag}", CLR_ACC if is_gk else CLR_DIM)

                if pen == GameState.PENALTY_NONE:
                    put(stdscr, row, 12, "OK", CLR_OK)
                else:
                    secs = m.secs_till_unpenalised[i] if i < len(m.secs_till_unpenalised) else 0
                    put(stdscr, row, 12,
                        f"PENALISED  {PENALTY_NAMES.get(pen, '?')}  ({secs}s)", CLR_CRIT)

                if is_me:
                    put(stdscr, row, 50, "← ME", CLR_ACC)
                row += 1
        row += 1

        # ══ MOTION LOCK ══
        row = section(stdscr, row, w, "MOTION LOCK (game_stop_lock)")
        lock_age = time.time() - last_lock_t if last_lock_t > 0 else -1.0
        if lock_val is None or lock_age < 0:
            put(stdscr, row, 4, "(game_stop_lock 토픽 없음 — lock 노드 미실행?)", CLR_WARN)
        elif lock_age > 2.0:
            put(stdscr, row, 4, f"STALE ({lock_age:.0f}s) — lock 노드 확인", CLR_CRIT)
        elif lock_val:
            put(stdscr, row, 4, "■ LOCKED  (자율 명령 차단 = 로봇 정지)", CLR_CRIT)
        else:
            put(stdscr, row, 4, "▶ FREE    (주행 허용)", CLR_OK)
        row += 1

        # ── 하단 범례 ──
        if h - 1 > row:
            put(stdscr, h - 1, 2,
                "PLAYING=green  READY=yellow  SET/INITIAL/FIN=red    Press Q to quit",
                CLR_DIM)

        stdscr.refresh()


def main(args=None):
    rclpy.init(args=args)
    node = GameStateMonitorNode()
    try:
        curses.wrapper(lambda stdscr: draw_dashboard(stdscr, node))
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
