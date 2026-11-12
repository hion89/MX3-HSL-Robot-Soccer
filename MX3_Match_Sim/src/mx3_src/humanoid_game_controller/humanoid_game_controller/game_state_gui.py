#!/usr/bin/env python3
"""
game_state_gui.py
=================

GameController 경기 상태를 순서도(상태 머신)로 그려서, 현재 상태 칸에
"불이 들어오게" 보여주는 PySide6 기반 GUI 모니터.

TUI(game_state_monitor_tui) 대비 한눈에 직관적으로 보이도록:
  - INITIAL → READY → SET → PLAYING → FINISHED 순서도에서 현재 상태가 점등
  - 큰 MOTION 배너: 정지(LOCKED, 빨강) / 주행(FREE, 초록)
  - 점수 · 시간 · phase/setplay
  - 우리 로봇별 패널티 상태(정상=초록, 패널티=빨강+남은초)

연결:
  구독: /game_state      (humanoid_behavior_interfaces/GameState)
        game_stop_lock   (std_msgs/Bool)

실행:
  ros2 run humanoid_game_controller game_state_gui
  # 내 로봇 강조:
  ros2 run humanoid_game_controller game_state_gui --ros-args -p player_number:=4
  (X11 forwarding 또는 NoMachine 등 GUI 환경 필요)

의존성:
  PySide6 (pose_slider_gui 와 동일):  pip3 install PySide6
"""
import sys
import time
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from humanoid_behavior_interfaces.msg import GameState

# PySide6 미설치 시 친절한 에러 (pose_slider_gui 와 동일 패턴)
try:
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QFrame,
    )
except ImportError as e:
    print("ERROR: PySide6 가 설치돼 있지 않습니다.")
    print("       pip3 install PySide6")
    print(f"       원인: {e}")
    sys.exit(1)


# ── enum 코드 → 이름 (RoboCupGameControlData.h 와 1:1) ──
# 순서도 칸 순서: state 코드 0~4 와 인덱스가 일치하도록 나열.
STATE_FLOW = ['INITIAL', 'READY', 'SET', 'PLAYING', 'FINISHED']
# 각 상태가 점등될 때의 색(의미 기반): 빨강=정지성, 노랑=준비, 초록=진행
STATE_COLOR = {
    0: '#607d8b',   # INITIAL  - 청회색
    1: '#f9a825',   # READY    - 앰버(포메이션 이동)
    2: '#e53935',   # SET      - 빨강(정지)
    3: '#43a047',   # PLAYING  - 초록(진행)
    4: '#546e7a',   # FINISHED - 회색
}
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
MAX_ROBOT_BOXES = 8   # 미리 만들어 두는 로봇 칸 수(players_per_team 만큼만 표시)


def fmt_time(secs: int) -> str:
    if secs < 0:
        return f"-{abs(secs) // 60:02d}:{abs(secs) % 60:02d}"
    return f"{secs // 60:02d}:{secs % 60:02d}"


# ────────────────────────────────────────────────────────────────────
# ROS 노드 — 최신 메시지만 보관(렌더는 Qt 타이머가 폴링)
# ────────────────────────────────────────────────────────────────────
class GameStateNode(Node):
    def __init__(self):
        super().__init__('game_state_gui')
        self.player_number = int(self.declare_parameter('player_number', 0).value)

        self._lock = threading.Lock()
        self._state_msg = None
        self._state_t = 0.0
        self._lock_val = None
        self._lock_t = 0.0

        self.create_subscription(GameState, '/game_state', self._cb_state, 10)
        self.create_subscription(Bool, 'game_stop_lock', self._cb_lock, 10)

    def _cb_state(self, msg):
        with self._lock:
            self._state_msg = msg
            self._state_t = time.time()

    def _cb_lock(self, msg):
        with self._lock:
            self._lock_val = msg.data
            self._lock_t = time.time()

    def snapshot(self):
        """(msg, msg_age, lock_val, lock_age) 스냅샷을 스레드 안전하게 반환."""
        with self._lock:
            now = time.time()
            msg_age = (now - self._state_t) if self._state_t > 0 else -1.0
            lock_age = (now - self._lock_t) if self._lock_t > 0 else -1.0
            return self._state_msg, msg_age, self._lock_val, lock_age


class RosSpinThread(threading.Thread):
    """rclpy 콜백 처리 전용 스레드 (pose_slider_gui 와 동일 패턴)."""
    def __init__(self, node):
        super().__init__(daemon=True)
        self.node = node
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set() and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)

    def stop(self):
        self._stop.set()


# ────────────────────────────────────────────────────────────────────
# 스타일 헬퍼
# ────────────────────────────────────────────────────────────────────
def style_state_box(label: QLabel, active: bool, color: str):
    """순서도 상태 칸 점등/소등 스타일."""
    if active:
        label.setStyleSheet(
            f"background:{color}; color:white; border-radius:8px;"
            "padding:10px 4px; font-size:15px; font-weight:600;"
            f"border:2px solid {color};"
        )
    else:
        # 소등: 옅은 회색 + 흐린 글씨
        label.setStyleSheet(
            "background:#eceff1; color:#b0bec5; border-radius:8px;"
            "padding:10px 4px; font-size:14px; font-weight:400;"
            "border:2px solid #eceff1;"
        )


def style_robot_box(label: QLabel, ok: bool, text: str):
    color_bg = '#e8f5e9' if ok else '#ffebee'
    color_fg = '#2e7d32' if ok else '#c62828'
    border = '#a5d6a7' if ok else '#ef9a9a'
    label.setStyleSheet(
        f"background:{color_bg}; color:{color_fg}; border:1px solid {border};"
        "border-radius:6px; padding:6px 8px; font-size:13px;"
    )
    label.setText(text)


# ────────────────────────────────────────────────────────────────────
# 메인 윈도우
# ────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self, node: GameStateNode):
        super().__init__()
        self.node = node
        self.setWindowTitle("KHUBER GameController Monitor")
        self.resize(820, 440)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(12)

        # ── 상단: 연결 상태 / 팀·패킷 ──
        top = QHBoxLayout()
        self.lbl_conn = QLabel("Waiting for /game_state ...")
        self.lbl_conn.setStyleSheet("font-size:13px; color:#f9a825;")
        self.lbl_meta = QLabel("")
        self.lbl_meta.setStyleSheet("font-size:13px; color:#90a4ae;")
        self.lbl_meta.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        top.addWidget(self.lbl_conn)
        top.addStretch(1)
        top.addWidget(self.lbl_meta)
        root.addLayout(top)

        # ── 상태 순서도 (점등) ──
        flow = QHBoxLayout()
        flow.setSpacing(6)
        self.state_boxes = []
        for i, name in enumerate(STATE_FLOW):
            box = QLabel(name)
            box.setAlignment(Qt.AlignCenter)
            box.setMinimumWidth(120)
            style_state_box(box, active=False, color=STATE_COLOR[i])
            self.state_boxes.append(box)
            flow.addWidget(box, 1)
            if i < len(STATE_FLOW) - 1:
                arrow = QLabel("→")
                arrow.setAlignment(Qt.AlignCenter)
                arrow.setStyleSheet("font-size:18px; color:#90a4ae;")
                flow.addWidget(arrow)
        root.addLayout(flow)

        # ── 중단: MOTION 배너 + phase/setplay ──
        mid = QHBoxLayout()
        mid.setSpacing(12)
        self.lbl_motion = QLabel("MOTION —")
        self.lbl_motion.setAlignment(Qt.AlignCenter)
        self.lbl_motion.setMinimumHeight(56)
        self.lbl_motion.setStyleSheet(
            "background:#cfd8dc; color:#37474f; border-radius:10px;"
            "font-size:18px; font-weight:600;")
        self.lbl_phase = QLabel("")
        self.lbl_phase.setAlignment(Qt.AlignCenter)
        self.lbl_phase.setMinimumHeight(56)
        self.lbl_phase.setStyleSheet(
            "background:#eceff1; color:#455a64; border-radius:10px; font-size:14px;")
        mid.addWidget(self.lbl_motion, 1)
        mid.addWidget(self.lbl_phase, 1)
        root.addLayout(mid)

        # ── 점수 / 시간 ──
        self.lbl_score = QLabel("")
        self.lbl_score.setAlignment(Qt.AlignCenter)
        self.lbl_score.setStyleSheet("font-size:20px; font-weight:600; color:#263238;")
        root.addWidget(self.lbl_score)

        # ── 구분선 ──
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color:#cfd8dc;")
        root.addWidget(line)

        # ── 우리 로봇 칸 ──
        root.addWidget(QLabel("<b>OUR ROBOTS</b>"))
        robots = QHBoxLayout()
        robots.setSpacing(8)
        self.robot_boxes = []
        for _ in range(MAX_ROBOT_BOXES):
            box = QLabel("")
            box.setAlignment(Qt.AlignCenter)
            box.setVisible(False)
            self.robot_boxes.append(box)
            robots.addWidget(box)
        robots.addStretch(1)
        root.addLayout(robots)

        root.addStretch(1)

        # ── 갱신 타이머 (10Hz 폴링) ──
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(100)

    # ----------------------------------------------------------------
    def refresh(self):
        m, msg_age, lock_val, lock_age = self.node.snapshot()

        # 연결 상태
        if m is None or msg_age < 0:
            self.lbl_conn.setText("Waiting for /game_state ...")
            self.lbl_conn.setStyleSheet("font-size:13px; color:#f9a825;")
            return
        if msg_age > 2.0:
            self.lbl_conn.setText(f"● STALE — {msg_age:.0f}s 무신호 (GC 끊김?)")
            self.lbl_conn.setStyleSheet("font-size:13px; color:#c62828; font-weight:600;")
        else:
            self.lbl_conn.setText(f"● Live ({msg_age:.1f}s)")
            self.lbl_conn.setStyleSheet("font-size:13px; color:#2e7d32; font-weight:600;")

        self.lbl_meta.setText(
            f"Team {m.our_team_number} · {COMPETITION_NAMES.get(m.competition_type, '?')} · "
            f"Packet #{m.packet_number}")

        # ── 순서도 점등: 현재 state 칸만 켜기 ──
        for i, box in enumerate(self.state_boxes):
            style_state_box(box, active=(i == m.state), color=STATE_COLOR[i])

        # ── MOTION 배너 ──
        if lock_val is None or lock_age < 0:
            self.lbl_motion.setText("MOTION  —  (lock 토픽 없음)")
            self.lbl_motion.setStyleSheet(
                "background:#cfd8dc; color:#37474f; border-radius:10px;"
                "font-size:16px; font-weight:600;")
        elif lock_age > 2.0:
            self.lbl_motion.setText("MOTION  —  STALE")
            self.lbl_motion.setStyleSheet(
                "background:#ffe0b2; color:#e65100; border-radius:10px;"
                "font-size:16px; font-weight:600;")
        elif lock_val:
            self.lbl_motion.setText("■  LOCKED  ·  로봇 정지")
            self.lbl_motion.setStyleSheet(
                "background:#e53935; color:white; border-radius:10px;"
                "font-size:18px; font-weight:600;")
        else:
            self.lbl_motion.setText("▶  FREE  ·  주행 허용")
            self.lbl_motion.setStyleSheet(
                "background:#43a047; color:white; border-radius:10px;"
                "font-size:18px; font-weight:600;")

        # ── phase / setplay ──
        kicking = "US" if m.we_are_kicking else "THEM"
        stopped = "  ·  STOPPED" if m.stopped else ""
        self.lbl_phase.setText(
            f"{PHASE_NAMES.get(m.game_phase, '?')}  ·  "
            f"SetPlay {SETPLAY_NAMES.get(m.set_play, '?')}\n"
            f"Kick {kicking}{stopped}")

        # ── 점수 / 시간 ──
        half = "1st" if m.first_half else "2nd"
        self.lbl_score.setText(
            f"US({m.our_team_number})  {m.our_score}  :  {m.opponent_score}  OPP"
            f"      ⏱ {fmt_time(m.secs_remaining)} ({half})   GK #{m.our_goalkeeper}")

        # ── 우리 로봇 칸 ──
        n = len(m.penalties)
        for i, box in enumerate(self.robot_boxes):
            if i >= n:
                box.setVisible(False)
                continue
            box.setVisible(True)
            player_no = i + 1
            gk = "GK " if player_no == m.our_goalkeeper else ""
            me = " ←ME" if (self.node.player_number and player_no == self.node.player_number) else ""
            pen = m.penalties[i]
            if pen == 0:
                style_robot_box(box, ok=True, text=f"#{player_no} {gk}OK{me}")
            else:
                secs = m.secs_till_unpenalised[i] if i < len(m.secs_till_unpenalised) else 0
                name = PENALTY_NAMES.get(pen, '?')
                style_robot_box(box, ok=False, text=f"#{player_no} {gk}{name} {secs}s{me}")


def main(args=None):
    rclpy.init(args=args)
    node = GameStateNode()
    spin = RosSpinThread(node)
    spin.start()

    app = QApplication(sys.argv)
    win = MainWindow(node)
    win.show()
    try:
        app.exec()
    finally:
        spin.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
