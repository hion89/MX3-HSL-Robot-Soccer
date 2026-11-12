#!/usr/bin/env python3
"""
motor_monitor_tui.py

KHUBER 모터 진단 TUI 모니터
/motor_diagnostics (std_msgs/String, JSON) 구독 → 터미널 대시보드 표시

실행:
    ros2 run humanoid_serial_bridge motor_monitor_tui

PWM, 전류, 온도를 컬러 바와 함께 실시간 표시합니다.
"""

import json
import curses
import time
import threading
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


# ── 표시 순서 (Left leg → Right leg → Left arm → Right arm) ──
DISPLAY_ORDER = [
    # Left leg
    "l_pelvis_pitch_joint",
    "l_pelvis_roll_joint",
    "l_pelvis_yaw_joint",
    "l_knee_pitch_joint",
    "l_ankle_pitch_joint",
    "l_ankle_roll_joint",
    # Right leg
    "r_pelvis_pitch_joint",
    "r_pelvis_roll_joint",
    "r_pelvis_yaw_joint",
    "r_knee_pitch_joint",
    "r_ankle_pitch_joint",
    "r_ankle_roll_joint",
    # Left arm
    "l_shoulder_pitch_joint",
    "l_shoulder_roll_joint",
    "l_elbow_pitch_joint",
    # Right arm
    "r_shoulder_pitch_joint",
    "r_shoulder_roll_joint",
    "r_elbow_pitch_joint",
]

# 축약 이름
SHORT_NAMES = {
    "l_pelvis_pitch_joint":   "L Hip Pitch ",
    "l_pelvis_roll_joint":    "L Hip Roll  ",
    "l_pelvis_yaw_joint":     "L Hip Yaw   ",
    "l_knee_pitch_joint":     "L Knee      ",
    "l_ankle_pitch_joint":    "L Ankle Pit ",
    "l_ankle_roll_joint":     "L Ankle Rol ",
    "r_pelvis_pitch_joint":   "R Hip Pitch ",
    "r_pelvis_roll_joint":    "R Hip Roll  ",
    "r_pelvis_yaw_joint":     "R Hip Yaw   ",
    "r_knee_pitch_joint":     "R Knee      ",
    "r_ankle_pitch_joint":    "R Ankle Pit ",
    "r_ankle_roll_joint":     "R Ankle Rol ",
    "l_shoulder_pitch_joint": "L Shldr Pit ",
    "l_shoulder_roll_joint":  "L Shldr Rol ",
    "l_elbow_pitch_joint":    "L Elbow     ",
    "r_shoulder_pitch_joint": "R Shldr Pit ",
    "r_shoulder_roll_joint":  "R Shldr Rol ",
    "r_elbow_pitch_joint":    "R Elbow     ",
}

# PWM 최대값 (MX-64AR: 885)
PWM_MAX = 885
# 전류 경고 임계값 (mA)
CURRENT_WARN_MA = 1500.0
CURRENT_CRIT_MA = 2500.0
# 온도 임계값 (°C)
TEMP_WARN = 50
TEMP_CRIT = 60


class MotorMonitorNode(Node):
    def __init__(self):
        super().__init__('motor_monitor_tui')
        self.latest_data: Optional[dict] = None
        self.data_lock = threading.Lock()
        self.last_recv_time = 0.0

        self.create_subscription(String, '/motor_diagnostics', self.cb_diag, 10)
        self.get_logger().info('Motor Monitor TUI started. Waiting for /motor_diagnostics ...')

    def cb_diag(self, msg: String):
        try:
            data = json.loads(msg.data)
            with self.data_lock:
                self.latest_data = data
                self.last_recv_time = time.time()
        except json.JSONDecodeError:
            pass


def make_bar(value: float, max_val: float, width: int = 12) -> str:
    """값에 비례하는 블록 바 생성."""
    ratio = min(abs(value) / max(max_val, 1e-6), 1.0)
    filled = int(ratio * width)
    empty = width - filled
    return "█" * filled + "░" * empty


def draw_dashboard(stdscr, node: MotorMonitorNode):
    """curses 메인 루프."""
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(200)  # 200ms refresh

    # 색상 초기화
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)    # 정상
    curses.init_pair(2, curses.COLOR_YELLOW, -1)   # 경고
    curses.init_pair(3, curses.COLOR_RED, -1)       # 위험
    curses.init_pair(4, curses.COLOR_CYAN, -1)      # 헤더
    curses.init_pair(5, curses.COLOR_WHITE, -1)     # 일반

    CLR_OK   = curses.color_pair(1)
    CLR_WARN = curses.color_pair(2)
    CLR_CRIT = curses.color_pair(3)
    CLR_HEAD = curses.color_pair(4) | curses.A_BOLD
    CLR_DIM  = curses.color_pair(5) | curses.A_DIM

    while True:
        # ROS 콜백 처리
        rclpy.spin_once(node, timeout_sec=0.0)

        key = stdscr.getch()
        if key == ord('q') or key == ord('Q'):
            break

        stdscr.erase()
        h, w = stdscr.getmaxyx()

        # ── 타이틀 ──
        title = " KHUBER Motor Diagnostics Monitor "
        stdscr.addstr(0, max(0, (w - len(title)) // 2), title, CLR_HEAD)

        with node.data_lock:
            data = node.latest_data
            last_t = node.last_recv_time

        # 연결 상태
        age = time.time() - last_t if last_t > 0 else -1
        if age < 0:
            status = "  Waiting for data ...  "
            stdscr.addstr(1, 2, status, CLR_WARN)
            stdscr.refresh()
            continue
        elif age > 3.0:
            status = f"  No data for {age:.0f}s — check bridge node  "
            stdscr.addstr(1, 2, status, CLR_CRIT)
        else:
            status = f"  Live ({age:.1f}s ago)  "
            stdscr.addstr(1, 2, status, CLR_OK)

        if data is None:
            stdscr.refresh()
            continue

        motors_list = data.get("motors", [])
        motors_by_name = {m["name"]: m for m in motors_list}

        # ── 헤더 행 ──
        row = 3
        header = (
            f" {'Joint':<16s} {'ID':>3s} │"
            f" {'PWM':>5s}  {'':12s} │"
            f" {'mA':>7s}  {'':12s} │"
            f" {'°C':>4s}  {'':12s} │"
            f" {'V':>5s}"
        )
        if row < h:
            stdscr.addstr(row, 1, header[:w-2], CLR_HEAD)
        row += 1
        if row < h:
            stdscr.addstr(row, 1, "─" * min(w - 2, 95), CLR_DIM)
        row += 1

        # ── 그룹 구분선 ──
        groups = [
            ("── Left Leg ──", 0),
            ("── Right Leg ──", 6),
            ("── Left Arm ──", 12),
            ("── Right Arm ──", 15),
        ]
        group_idx = {g[1]: g[0] for g in groups}

        for i, joint_name in enumerate(DISPLAY_ORDER):
            if row >= h - 2:
                break

            # 그룹 헤더
            if i in group_idx:
                stdscr.addstr(row, 2, group_idx[i], CLR_DIM)
                row += 1
                if row >= h - 2:
                    break

            m = motors_by_name.get(joint_name)
            if m is None:
                stdscr.addstr(row, 2, f" {SHORT_NAMES.get(joint_name, joint_name):<16s}  --", CLR_DIM)
                row += 1
                continue

            short = SHORT_NAMES.get(joint_name, joint_name[:16])
            dxl_id = m["id"]
            pwm = m["pwm"]
            cur_ma = m["current_ma"]
            temp = m["temp_c"]
            volt = m["voltage_v"]

            # PWM 색상
            pwm_ratio = abs(pwm) / PWM_MAX
            pwm_clr = CLR_CRIT if pwm_ratio > 0.8 else (CLR_WARN if pwm_ratio > 0.5 else CLR_OK)

            # 전류 색상
            cur_abs = abs(cur_ma)
            cur_clr = CLR_CRIT if cur_abs > CURRENT_CRIT_MA else (CLR_WARN if cur_abs > CURRENT_WARN_MA else CLR_OK)

            # 온도 색상
            temp_clr = CLR_CRIT if temp >= TEMP_CRIT else (CLR_WARN if temp >= TEMP_WARN else CLR_OK)

            # 라인 구성
            try:
                stdscr.addstr(row, 1, f" {short:<16s}", curses.color_pair(5))
                stdscr.addstr(f"{dxl_id:>3d}", CLR_DIM)
                stdscr.addstr(" │ ")
                stdscr.addstr(f"{pwm:>+5d}", pwm_clr)
                stdscr.addstr(f"  {make_bar(pwm, PWM_MAX)}", pwm_clr)
                stdscr.addstr(" │ ")
                stdscr.addstr(f"{cur_ma:>+7.1f}", cur_clr)
                stdscr.addstr(f"  {make_bar(cur_ma, CURRENT_CRIT_MA)}", cur_clr)
                stdscr.addstr(" │ ")
                stdscr.addstr(f"{temp:>4d}", temp_clr)
                stdscr.addstr(f"  {make_bar(temp, 80)}", temp_clr)
                stdscr.addstr(" │ ")
                stdscr.addstr(f"{volt:>5.1f}")
            except curses.error:
                pass

            row += 1

        # ── 하단 범례 ──
        row += 1
        if row < h - 1:
            try:
                stdscr.addstr(row, 2, " PWM: ", CLR_DIM)
                stdscr.addstr("<50%", CLR_OK)
                stdscr.addstr("  ")
                stdscr.addstr("50-80%", CLR_WARN)
                stdscr.addstr("  ")
                stdscr.addstr(">80%", CLR_CRIT)
                stdscr.addstr("   │  TEMP: ", CLR_DIM)
                stdscr.addstr(f"<{TEMP_WARN}", CLR_OK)
                stdscr.addstr("  ")
                stdscr.addstr(f"{TEMP_WARN}-{TEMP_CRIT}", CLR_WARN)
                stdscr.addstr("  ")
                stdscr.addstr(f">{TEMP_CRIT}", CLR_CRIT)
                stdscr.addstr("   │  Press Q to quit", CLR_DIM)
            except curses.error:
                pass

        stdscr.refresh()


def main(args=None):
    rclpy.init(args=args)
    node = MotorMonitorNode()

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