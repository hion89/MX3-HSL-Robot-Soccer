#!/usr/bin/env python3
"""
keyframe_editor_gui.py
======================

standup / kick 키프레임 YAML 을 phase 단위로 편집하고, 실로봇에서 바로
재생해보는 PySide6 GUI.

pose_slider_gui 가 "자세 하나를 만들어 YAML 조각을 클립보드로 뽑는" 도구라면,
이건 "시퀀스 전체를 열어 phase 를 오가며 고치고 파일에 직접 저장하는" 도구다.
넘어졌다 일어나는 동작처럼 phase 가 5~6개 이어지는 시퀀스는 조각을 하나씩
복사·붙여넣기로 다루기엔 너무 번거롭다.

────────────────────────────────────────────────────────────────────
워크플로 (뒤로 넘어진 상태 튜닝 예시)
────────────────────────────────────────────────────────────────────
  0) 로봇을 실제로 뒤로 눕힌다.
     ★ fall_manager 가 자동으로 standup 을 걸면 편집이 불가능하다.
       반드시 먼저 내릴 것:  (fall_manager_node 를 Ctrl+C)
  1) Sequence = backward 선택 → phase 목록에서 'base' 클릭
  2) [Go To Phase] → 로봇이 그 자세로 감. 실제로 어떻게 되는지 눈으로 본다
  3) 슬라이더로 수정 → [Apply] → 다시 눈으로 확인. 반복
     · 손으로 잡아보고 싶으면 [Torque OFF] → 자세 잡기 → [Sync from Robot]
     · 좌우 대칭은 [Mirror R→L] 한 번이면 된다
  4) 마음에 들면 [Save Phase] (편집 버퍼에 반영)
  5) 다음 phase 로 이동해 2~4 반복
  6) [Save YAML] → 파일 저장 → [Reload on Robot] → 노드가 새 YAML 을 읽음
  7) [Play Sequence] 로 처음부터 통째로 돌려본다

────────────────────────────────────────────────────────────────────
연결
────────────────────────────────────────────────────────────────────
  발행 : /joint_goals                    (JointState)  Apply
  구독 : /joint_states                   (JointState)  실제 위치 / Sync
  구독 : /u2d2/behavior_status           (String,JSON) 재생 상태 표시
  서비스: /u2d2/torque_enable            토크 토글
  서비스: /u2d2/set_joint_profile_velocity 는 쓰지 않는다
          (phase rpm 은 Apply 시 /u2d2/set_profile_velocity 로 일괄 적용)
  서비스: /u2d2/standup                  goto / next / back / reload / start / stop
  서비스: /u2d2/kick                     goto / next / reload / start / stop

실행:
  ros2 run humanoid_serial_bridge keyframe_editor_gui
  ros2 run humanoid_serial_bridge keyframe_editor_gui --ros-args \
      -p yaml_path:=/절대/경로/standup_keyframes.yaml

의존성:
  pip3 install PySide6
"""
import copy
import json
import os
import shutil
import sys
import threading
import time

import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, String

from std_srvs.srv import Trigger

from humanoid_behavior_interfaces.srv import (
    StandUp, TorqueEnable, Kick, GetJointLimits,
)

try:
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QGridLayout, QGroupBox, QLabel, QSlider, QPushButton,
        QDoubleSpinBox, QListWidget, QComboBox, QMessageBox, QSplitter,
        QInputDialog, QPlainTextEdit,
    )
except ImportError as e:
    print("ERROR: PySide6 가 설치돼 있지 않습니다.")
    print("       pip3 install PySide6")
    print(f"       원인: {e}")
    sys.exit(1)


# ────────────────────────────────────────────────────────────────────
# 조인트 정의 — YAML 의 joints 블록에 들어가는 순서 그대로 유지한다.
# 순서를 바꾸면 diff 가 지저분해져서 리뷰가 어려워진다.
# ────────────────────────────────────────────────────────────────────
JOINT_ORDER = [
    "r_pelvis_pitch_joint", "r_pelvis_roll_joint", "r_pelvis_yaw_joint",
    "r_knee_pitch_joint", "r_ankle_pitch_joint", "r_ankle_roll_joint",
    "l_pelvis_pitch_joint", "l_pelvis_roll_joint", "l_pelvis_yaw_joint",
    "l_knee_pitch_joint", "l_ankle_pitch_joint", "l_ankle_roll_joint",
    "r_shoulder_pitch_joint", "r_shoulder_roll_joint", "r_elbow_pitch_joint",
    "l_shoulder_pitch_joint", "l_shoulder_roll_joint", "l_elbow_pitch_joint",
]

# 좌우 대칭 쌍. Mirror 버튼이 쓴다.
#   기존 키프레임을 보면 좌우가 부호 반전 관계다 (r=+0.42 ↔ l=-0.42).
#   전 관절이 같은 규칙이라 별도 예외 테이블이 필요 없다.
MIRROR_PAIRS = [
    ("r_pelvis_pitch_joint", "l_pelvis_pitch_joint"),
    ("r_pelvis_roll_joint", "l_pelvis_roll_joint"),
    ("r_pelvis_yaw_joint", "l_pelvis_yaw_joint"),
    ("r_knee_pitch_joint", "l_knee_pitch_joint"),
    ("r_ankle_pitch_joint", "l_ankle_pitch_joint"),
    ("r_ankle_roll_joint", "l_ankle_roll_joint"),
    ("r_shoulder_pitch_joint", "l_shoulder_pitch_joint"),
    ("r_shoulder_roll_joint", "l_shoulder_roll_joint"),
    ("r_elbow_pitch_joint", "l_elbow_pitch_joint"),
]

# 편집 중 Goal PWM 을 100% 로 올릴 대상.
#   motor_gains.yaml 의 arms 그룹과 같은 관절들. 다리/목은 원래 상한이
#   안 걸려 있으므로(YAML 에 goal_pwm 미지정) 건드릴 필요가 없다.
FULL_PWM_JOINTS = JOINT_ORDER[12:]

SLIDER_SCALE = 1000          # QSlider 는 int 만 → rad × 1000 (한 칸 0.001 rad)
# 슬라이더 초기 범위 ±rad. 로봇이 안 붙어 있을 때(오프라인 편집)만 쓰인다.
# 로봇이 붙으면 /u2d2/get_joint_limits 응답으로 관절마다 실제 한계로 교체된다.
#   ★ 하드코딩 범위를 그대로 두면 안 되는 이유:
#     send_goals() 는 servo_map 의 전역 tick 범위로만 clamp 하고 SyncWrite 하며,
#     SyncWrite 는 응답이 없다. 모터 EEPROM 한계를 벗어난 목표는 조용히 거부되어
#     "슬라이더는 움직이는데 관절은 안 움직이는" 구간이 생긴다.
JOINT_RANGE_FALLBACK = 3.20


# ────────────────────────────────────────────────────────────────────
# ROS 노드 — GUI 스레드와 분리
# ────────────────────────────────────────────────────────────────────
class RosNode(Node):
    """GUI 는 이 노드의 메서드만 호출한다. Qt 위젯은 절대 건드리지 않는다.

    콜백은 lock 으로 보호된 캐시에만 쓰고, GUI 쪽 QTimer 가 주기적으로 읽는다.
    (ROS 콜백 스레드에서 위젯을 만지면 Qt thread-safety 위반이다)
    """

    def __init__(self):
        super().__init__("keyframe_editor_gui")

        self.declare_parameter("yaml_path", "")
        self.declare_parameter("mode", "standup")   # standup | kick

        self.pub_goals = self.create_publisher(JointState, "/joint_goals", 10)
        self.pub_speed = self.create_publisher(
            Float32, "/u2d2/set_profile_velocity", 10)
        # 팔 Goal PWM 상한 승격용. motor_gains.yaml 의 운용값(예: 100 = 11.3%)
        # 으로는 팔이 목표까지 이동하지 못한다(실측 이동 하한 150).
        self.pub_goal_pwm = self.create_publisher(
            JointState, "/u2d2/set_joint_goal_pwm", 10)

        self.create_subscription(
            JointState, "/joint_states", self._on_joint_states, 10)
        self.create_subscription(
            String, "/u2d2/behavior_status", self._on_behavior_status, 10)

        self.cli_torque = self.create_client(TorqueEnable, "/u2d2/torque_enable")
        self.cli_standup = self.create_client(StandUp, "/u2d2/standup")
        self.cli_kick = self.create_client(Kick, "/u2d2/kick")
        # PWM 상한을 YAML 운용값으로 되돌릴 때 쓴다(게인도 함께 재적용).
        self.cli_reload_gains = self.create_client(
            Trigger, "/body_motor_controller/reload_gains")
        # 슬라이더 범위를 모터 EEPROM Min/Max Position Limit 에 맞추기 위해 사용.
        #   ★ 왜 반드시 필요한가:
        #     send_goals() 는 servo_map 의 전역 tick 범위(0~4095)로만 clamp 하고
        #     SyncWrite 한다. 모터별 EEPROM 한계는 보지 않는다. 그리고 SyncWrite
        #     는 응답이 없으므로, 한계 밖 목표는 모터가 조용히 거부하고 끝난다.
        #     → 슬라이더에 하드코딩 범위를 쓰면 "움직이지 않는 구간" 이 생기고
        #       사용자는 이유를 알 수 없다.
        self.cli_limits = self.create_client(
            GetJointLimits, "/u2d2/get_joint_limits")

        self._lock = threading.Lock()
        self._positions = {}
        self._behavior = {}
        # {joint_name: (lower_rad, upper_rad)}. 서비스 응답이 오면 채워진다.
        self._limits = {}
        self._limits_msg = ""
        self._limits_new = False

        self.get_logger().info("keyframe_editor_gui ROS node ready")

    # ── 콜백 ──
    def _on_joint_states(self, msg: JointState):
        with self._lock:
            for n, p in zip(msg.name, msg.position):
                self._positions[n] = float(p)

    def _on_behavior_status(self, msg: String):
        try:
            data = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        with self._lock:
            self._behavior = data

    # ── GUI 가 호출 ──
    def get_positions(self) -> dict:
        with self._lock:
            return dict(self._positions)

    def get_behavior(self) -> dict:
        with self._lock:
            return dict(self._behavior)

    def publish_goals(self, joint_dict: dict):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(joint_dict.keys())
        msg.position = [float(v) for v in joint_dict.values()]
        self.pub_goals.publish(msg)

    def publish_speed(self, rpm: float):
        """Apply 직전에 부른다.

        ★ 순서가 중요하다. 다이나믹셀은 Goal Position 을 쓰는 '그 순간의'
          Profile Velocity 로 이동을 시작한다. 목표를 먼저 보내면 직전 속도로
          출발해버린다. 그래서 속도 → (한 사이클 대기) → 목표 순으로 보낸다.
        """
        self.pub_speed.publish(Float32(data=float(rpm)))

    def request_limits(self):
        """관절 한계를 비동기로 요청한다. 응답은 get_limits() 로 꺼낸다."""
        if not self.cli_limits.service_is_ready():
            with self._lock:
                self._limits_msg = ("/u2d2/get_joint_limits 준비 안 됨 — "
                                    "슬라이더 범위는 기본값 유지")
                self._limits_new = True
            return
        fut = self.cli_limits.call_async(GetJointLimits.Request())
        fut.add_done_callback(self._on_limits)

    def _on_limits(self, fut):
        try:
            res = fut.result()
        except Exception as e:
            with self._lock:
                self._limits_msg = f"관절 한계 조회 실패: {e}"
                self._limits_new = True
            return
        lim = {}
        for n, lo, hi in zip(res.joint_names, res.lower_limits_rad,
                             res.upper_limits_rad):
            lim[str(n)] = (float(lo), float(hi))
        with self._lock:
            self._limits = lim
            self._limits_msg = f"관절 한계 {len(lim)}개 수신: {res.message}"
            self._limits_new = True

    def get_limits(self):
        """(limits, message, is_new) 스냅샷. is_new 는 한 번만 True."""
        with self._lock:
            new = self._limits_new
            self._limits_new = False
            return dict(self._limits), self._limits_msg, new

    def publish_goal_pwm(self, joint_names, value: float = 0.0):
        """관절별 Goal PWM 상한을 기입한다.

        value <= 0 이면 노드가 그 모터의 PWM Limit(=100%)으로 해석한다.
        모터별 PWM Limit 을 GUI 가 몰라도 "최대로" 를 표현할 수 있다.
        """
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(joint_names)
        msg.effort = [float(value)] * len(msg.name)
        self.pub_goal_pwm.publish(msg)

    def call_reload_gains(self, done_cb=None):
        """게인/Goal PWM 을 YAML 값으로 재적용 (= PWM 상한 원복)."""
        if not self.cli_reload_gains.service_is_ready():
            if done_cb:
                done_cb(False, "/body_motor_controller/reload_gains 준비 안 됨")
            return
        fut = self.cli_reload_gains.call_async(Trigger.Request())
        if done_cb:
            fut.add_done_callback(lambda f: _safe_done(f, done_cb))

    def call_torque(self, enable: bool, done_cb=None):
        if not self.cli_torque.service_is_ready():
            if done_cb:
                done_cb(False, "/u2d2/torque_enable 준비 안 됨")
            return
        req = TorqueEnable.Request()
        req.enable = bool(enable)
        fut = self.cli_torque.call_async(req)
        if done_cb:
            fut.add_done_callback(
                lambda f: _safe_done(f, done_cb))

    def call_standup(self, command: str, fall_direction: str = "",
                     done_cb=None):
        if not self.cli_standup.service_is_ready():
            if done_cb:
                done_cb(False, "/u2d2/standup 준비 안 됨")
            return
        req = StandUp.Request()
        req.command = command
        req.fall_direction = fall_direction
        # 0 을 넣으면 노드가 기본값으로 대체한다(`req.final_rpm or 200.0`).
        req.final_rpm = 0.0
        req.pos_tolerance_rad = 0.0
        req.settle_count = 0
        req.timeout_sec = 0.0
        fut = self.cli_standup.call_async(req)
        if done_cb:
            fut.add_done_callback(lambda f: _safe_done(f, done_cb))

    def call_kick(self, command: str, leg: str = "right", done_cb=None):
        if not self.cli_kick.service_is_ready():
            if done_cb:
                done_cb(False, "/u2d2/kick 준비 안 됨")
            return
        req = Kick.Request()
        req.command = command
        req.leg = leg
        req.final_rpm = 0.0
        req.timeout_sec = 0.0
        fut = self.cli_kick.call_async(req)
        if done_cb:
            fut.add_done_callback(lambda f: _safe_done(f, done_cb))


def _safe_done(fut, cb):
    """서비스 응답을 GUI 콜백으로 넘긴다. 예외는 메시지로 바꾼다."""
    try:
        res = fut.result()
        cb(bool(res.success), str(res.message))
    except Exception as e:
        cb(False, f"exception: {e}")


class RosSpinThread(threading.Thread):
    def __init__(self, node):
        super().__init__(daemon=True)
        self.node = node
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set() and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def stop(self):
        self._stop.set()
        self.join(timeout=1.0)


# ────────────────────────────────────────────────────────────────────
# 키프레임 파일 모델 — YAML 구조를 그대로 들고 있는다
# ────────────────────────────────────────────────────────────────────
class KeyframeDoc:
    """YAML 전체를 메모리에 들고, phase 단위로 읽고 쓴다.

    파일에 없는 키(주석 등)는 PyYAML 이 보존하지 못한다. 그래서 저장 시
    원본을 .bak 으로 남긴다 — 헤더 주석을 잃었을 때 되살릴 수 있게.
    """

    def __init__(self):
        self.path = ""
        self.data = {}
        self.seq_keys = []
        self.dirty = False

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("최상위가 dict 가 아닙니다")

        # 시퀀스 키 자동 판정: 값이 'phase 리스트' 인 최상위 키를 모두 찾는다.
        # standup 은 forward_sequence/backward_sequence, kick 은 다른 이름일
        # 수 있으므로 이름을 하드코딩하지 않는다.
        seq_keys = []
        for k, v in data.items():
            if isinstance(v, list) and v and isinstance(v[0], dict) \
                    and "joints" in v[0]:
                seq_keys.append(k)
        if not seq_keys:
            raise ValueError(
                "phase 시퀀스를 못 찾았습니다 "
                "(각 항목에 'joints' 가 있는 리스트가 최상위에 있어야 함)")

        self.path = path
        self.data = data
        self.seq_keys = seq_keys
        self.dirty = False

    def phases(self, seq_key: str) -> list:
        return self.data.get(seq_key, [])

    def phase_names(self, seq_key: str) -> list:
        return [str(p.get("name", f"phase{i}"))
                for i, p in enumerate(self.phases(seq_key))]

    def get_phase(self, seq_key: str, idx: int) -> dict:
        return self.phases(seq_key)[idx]

    def set_phase(self, seq_key: str, idx: int, phase: dict):
        self.phases(seq_key)[idx] = phase
        self.dirty = True

    def insert_phase(self, seq_key: str, idx: int, phase: dict):
        self.phases(seq_key).insert(idx, phase)
        self.dirty = True

    def remove_phase(self, seq_key: str, idx: int):
        del self.phases(seq_key)[idx]
        self.dirty = True

    def move_phase(self, seq_key: str, idx: int, delta: int) -> int:
        seq = self.phases(seq_key)
        j = idx + delta
        if not (0 <= j < len(seq)):
            return idx
        seq[idx], seq[j] = seq[j], seq[idx]
        self.dirty = True
        return j

    def save(self, path: str = "") -> str:
        """YAML 저장. 원본은 .bak 으로 백업한다.

        ★ PyYAML 은 주석을 보존하지 못한다. 원본 파일 상단의 설명 주석이
          사라지므로, 되살릴 수 있게 반드시 백업을 남긴다.
        """
        path = path or self.path
        if os.path.isfile(path):
            shutil.copy2(path, path + ".bak")
        # joints 안의 관절 순서를 JOINT_ORDER 로 정렬해 diff 를 안정화한다.
        out = copy.deepcopy(self.data)
        for k in self.seq_keys:
            for ph in out.get(k, []):
                j = ph.get("joints", {})
                ph["joints"] = {n: float(j[n]) for n in JOINT_ORDER if n in j}
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False,
                           default_flow_style=False)
        self.path = path
        self.dirty = False
        return path


def diagnose_save_path(path: str) -> dict:
    """저장 경로가 안전한지 진단한다.

    ★ 왜 필요한가:
      노드는 launch 가 넘긴 share 경로(install/.../share/...)를 읽는다.
      reload 가 그 파일을 다시 읽으므로 GUI 도 거기에 저장해야 반영된다.
      그런데 그 파일이 '복사본' 이면 다음 colcon build 때 소스로 덮여
      편집 내용이 통째로 날아간다.
      --symlink-install 이면 심볼릭 링크라 소스와 같은 파일이므로 안전하다.

    반환: {"real": 실제경로, "is_link": bool, "in_install": bool,
           "risky": bool, "src_guess": str}
    """
    real = os.path.realpath(path)
    is_link = os.path.islink(path)
    in_install = f"{os.sep}install{os.sep}" in os.path.abspath(path)
    # install 안인데 링크가 아니면 = 복사본 → 재빌드 시 덮어써진다
    risky = in_install and not is_link and f"{os.sep}install{os.sep}" in real

    src_guess = ""
    if risky:
        # 워크스페이스 루트(install 과 src 를 함께 가진 디렉터리)를 찾아
        # src 아래에서 같은 이름의 파일을 best-effort 로 찾는다.
        d = os.path.abspath(path)
        while d != os.path.dirname(d):
            d = os.path.dirname(d)
            if os.path.basename(d) == "install":
                ws = os.path.dirname(d)
                src = os.path.join(ws, "src")
                if os.path.isdir(src):
                    base = os.path.basename(path)
                    for root, _dirs, files in os.walk(src):
                        if base in files:
                            src_guess = os.path.join(root, base)
                            break
                break
    return {"real": real, "is_link": is_link, "in_install": in_install,
            "risky": risky, "src_guess": src_guess}


def default_yaml_path() -> str:
    """설치된 share 안의 standup_keyframes.yaml 경로.

    --symlink-install 로 빌드했다면 이 경로는 소스의 config/ 를 가리키는
    심볼릭 링크다. 즉 여기에 저장하면 소스 트리도 함께 갱신되고,
    노드의 reload 도 같은 파일을 다시 읽는다.
    """
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(
            get_package_share_directory("humanoid_serial_bridge"),
            "config", "standup_keyframes.yaml")
    except Exception:
        return ""


# ────────────────────────────────────────────────────────────────────
# 메인 윈도우
# ────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self, ros: RosNode):
        super().__init__()
        self.ros = ros
        self.doc = KeyframeDoc()
        self.cur_seq = ""
        self.cur_idx = -1
        self._loading = False      # 슬라이더 프로그램적 변경 중 시그널 무시용
        self._pending_apply = None  # 속도→목표 순서 지연용
        # 팔 Goal PWM 을 100% 로 유지하는 모드.
        #   ★ 왜 계속 재기입하나:
        #     standup/kick behavior 는 종료(_finish)할 때 PWM 을 YAML 운용값
        #     으로 되돌린다. Go To Phase / Play Sequence 를 쓰면 그 시점에
        #     팔 상한이 다시 100(11.3%)으로 떨어져 Apply 가 안 먹는다.
        #     1Hz 로 다시 써서 편집 중에는 항상 100% 를 유지한다.
        #     (2바이트 SyncWrite 1회/초 — 비용은 무시할 수준)
        self._full_pwm = False
        # {joint_name: (lo, hi)} 실제 모터 한계. 비어 있으면 fallback 범위 사용.
        self._limits = {}

        # ★ 노드가 들고 있는 키프레임이 편집기 내용과 다른 상태인가.
        #   Save Phase 는 '메모리 버퍼' 만 바꾼다. Play/Goto 는 노드가 기동 시
        #   (또는 마지막 reload 때) 읽어둔 시퀀스를 재생하므로, 파일 저장 +
        #   reload 를 거치지 않으면 편집 전 동작이 그대로 나온다.
        #   사용자가 이 순서를 기억하게 만들면 안 되므로 여기서 추적한다.
        self._robot_stale = False
        # reload 응답(ROS 스레드) 후 GUI 스레드에서 이어서 할 일
        self._after_reload = None

        self.setWindowTitle("KHUBER Keyframe Editor")
        self.resize(1280, 860)
        self._build_ui()

        # 실제 위치 / behavior 상태 갱신
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(100)

        # 팔 PWM 100% 재기입 타이머 (behavior 종료 시 원복되는 것을 덮어씀)
        self.pwm_timer = QTimer(self)
        self.pwm_timer.timeout.connect(self._reassert_full_pwm)
        self.pwm_timer.start(1000)

        # 편집 도구는 팔에 힘이 들어가야 쓸 수 있으므로 기본 ON 으로 시작한다.
        self._set_full_pwm(True, quiet=False)

        # 관절 한계 조회. 노드가 늦게 뜰 수 있으므로 응답이 올 때까지 재시도한다.
        self._limit_retry = 0
        self.ros.request_limits()
        self.limit_timer = QTimer(self)
        self.limit_timer.timeout.connect(self._retry_limits)
        self.limit_timer.start(2000)

        path = str(ros.get_parameter("yaml_path").value) or default_yaml_path()
        if path and os.path.isfile(path):
            self._load_path(path)
        else:
            self._log("YAML 을 열어주세요 ([Open YAML])")

    # ────────────────────────────────────────────────────────────
    # UI 구성
    # ────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)

        # ── 상단: 파일 / 시퀀스 ──
        top = QHBoxLayout()
        self.btn_open = QPushButton("Open YAML")
        self.btn_open.clicked.connect(self._on_open)
        self.lbl_path = QLabel("(파일 없음)")
        self.lbl_path.setStyleSheet("color:#555;")
        self.cmb_seq = QComboBox()
        self.cmb_seq.currentTextChanged.connect(self._on_seq_changed)
        top.addWidget(self.btn_open)
        top.addWidget(QLabel("Sequence:"))
        top.addWidget(self.cmb_seq)
        top.addWidget(self.lbl_path, 1)
        outer.addLayout(top)

        split = QSplitter(Qt.Horizontal)
        outer.addWidget(split, 1)

        # ── 좌: phase 목록 ──
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(QLabel("Phases"))
        self.lst = QListWidget()
        self.lst.currentRowChanged.connect(self._on_phase_selected)
        lv.addWidget(self.lst, 1)

        row = QGridLayout()
        for i, (txt, fn) in enumerate([
            ("Add", self._on_add), ("Duplicate", self._on_dup),
            ("Delete", self._on_del), ("Rename", self._on_rename),
            ("Move ↑", lambda: self._on_move(-1)),
            ("Move ↓", lambda: self._on_move(1)),
        ]):
            b = QPushButton(txt)
            b.clicked.connect(fn)
            row.addWidget(b, i // 2, i % 2)
        lv.addLayout(row)
        split.addWidget(left)

        # ── 중: 슬라이더 ──
        mid = QWidget()
        mv = QVBoxLayout(mid)

        meta = QHBoxLayout()
        meta.addWidget(QLabel("rpm:"))
        self.spn_rpm = QDoubleSpinBox()
        self.spn_rpm.setRange(1.0, 45.0)     # Velocity Limit 45.3rpm 이내
        self.spn_rpm.setSingleStep(5.0)
        self.spn_rpm.setValue(30.0)
        meta.addWidget(self.spn_rpm)
        meta.addWidget(QLabel("hold_sec:"))
        self.spn_hold = QDoubleSpinBox()
        self.spn_hold.setRange(0.0, 10.0)
        self.spn_hold.setSingleStep(0.1)
        self.spn_hold.setValue(0.5)
        meta.addWidget(self.spn_hold)
        meta.addStretch(1)
        mv.addLayout(meta)

        self.sliders = {}
        self.val_labels = {}
        self.cur_labels = {}
        self.lim_labels = {}
        for title, names in (
            ("Legs", JOINT_ORDER[:12]),
            ("Arms", JOINT_ORDER[12:]),
        ):
            box = QGroupBox(title)
            g = QGridLayout(box)
            g.addWidget(QLabel("joint"), 0, 0)
            g.addWidget(QLabel("target"), 0, 2)
            g.addWidget(QLabel("actual"), 0, 3)
            g.addWidget(QLabel("limit"), 0, 4)
            for r, name in enumerate(names, start=1):
                g.addWidget(QLabel(name.replace("_joint", "")), r, 0)
                s = QSlider(Qt.Horizontal)
                s.setRange(int(-JOINT_RANGE_FALLBACK * SLIDER_SCALE),
                           int(JOINT_RANGE_FALLBACK * SLIDER_SCALE))
                s.valueChanged.connect(self._on_slider)
                g.addWidget(s, r, 1)
                vl = QLabel("0.000")
                vl.setMinimumWidth(56)
                vl.setFont(QFont("monospace"))
                g.addWidget(vl, r, 2)
                cl = QLabel("--")
                cl.setMinimumWidth(56)
                cl.setFont(QFont("monospace"))
                cl.setStyleSheet("color:#0a7;")
                g.addWidget(cl, r, 3)
                ll = QLabel("--")
                ll.setMinimumWidth(112)
                ll.setFont(QFont("monospace"))
                ll.setStyleSheet("color:#888;")
                g.addWidget(ll, r, 4)
                self.sliders[name] = s
                self.val_labels[name] = vl
                self.cur_labels[name] = cl
                self.lim_labels[name] = ll
            mv.addWidget(box)
        mv.addStretch(1)
        split.addWidget(mid)

        # ── 우: 동작 버튼 ──
        right = QWidget()
        rv = QVBoxLayout(right)

        gb_edit = QGroupBox("편집")
        ge = QVBoxLayout(gb_edit)
        for txt, fn, tip in [
            ("Sync from Robot", self._on_sync,
             "현재 실제 관절 위치를 슬라이더로 가져온다 (토크 OFF 후 손으로 잡고 사용)"),
            ("Mirror R→L", lambda: self._on_mirror(True),
             "오른쪽 값을 부호 반전해 왼쪽에 복사"),
            ("Mirror L→R", lambda: self._on_mirror(False),
             "왼쪽 값을 부호 반전해 오른쪽에 복사"),
            ("Save Phase", self._on_save_phase,
             "슬라이더 값을 이 phase 에 반영 (아직 파일에는 안 씀)"),
            ("Revert Phase", self._on_revert_phase,
             "이 phase 를 마지막 저장 상태로 되돌린다"),
        ]:
            b = QPushButton(txt)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            ge.addWidget(b)
        rv.addWidget(gb_edit)

        gb_robot = QGroupBox("로봇")
        gr = QVBoxLayout(gb_robot)
        self.btn_apply = QPushButton("Apply (슬라이더 → 로봇)")
        self.btn_apply.setStyleSheet("font-weight:bold;")
        self.btn_apply.clicked.connect(self._on_apply)
        gr.addWidget(self.btn_apply)
        for txt, fn in [
            ("Go To Phase", self._on_goto),
            ("Next Phase", lambda: self._on_seq_cmd("next")),
            ("Back Phase", lambda: self._on_seq_cmd("back")),
            ("Play Sequence", self._on_play),
            ("STOP", lambda: self._on_seq_cmd("stop")),
        ]:
            b = QPushButton(txt)
            if txt == "STOP":
                b.setStyleSheet("background:#c33; color:white; font-weight:bold;")
            b.clicked.connect(fn)
            gr.addWidget(b)
        # ── 팔 PWM 상한 ──
        self.btn_pwm = QPushButton("Arms PWM: --")
        self.btn_pwm.setToolTip(
            "편집 중에는 팔 Goal PWM 을 100% 로 올려야 한다.\n"
            "motor_gains.yaml 의 운용값(예: 100 = 11.3%)으로는\n"
            "팔이 목표까지 이동하지 못한다 (실측 이동 하한 150).")
        self.btn_pwm.clicked.connect(self._on_toggle_pwm)
        gr.addWidget(self.btn_pwm)

        tq = QHBoxLayout()
        b_on = QPushButton("Torque ON")
        b_on.clicked.connect(lambda: self._on_torque(True))
        b_off = QPushButton("Torque OFF")
        b_off.clicked.connect(lambda: self._on_torque(False))
        tq.addWidget(b_on)
        tq.addWidget(b_off)
        gr.addLayout(tq)
        rv.addWidget(gb_robot)

        gb_file = QGroupBox("파일")
        gf = QVBoxLayout(gb_file)
        b_save = QPushButton("Save YAML")
        b_save.setStyleSheet("font-weight:bold;")
        b_save.clicked.connect(self._on_save_yaml)
        gf.addWidget(b_save)
        b_reload = QPushButton("Reload on Robot")
        b_reload.setToolTip("노드가 YAML 을 다시 읽게 한다 (Save YAML 후에 누를 것)")
        b_reload.clicked.connect(self._on_reload)
        gf.addWidget(b_reload)
        self.lbl_stale = QLabel("로봇 반영: 최신")
        self.lbl_stale.setStyleSheet("color:#2a6;")
        gf.addWidget(self.lbl_stale)
        rv.addWidget(gb_file)

        self.lbl_status = QLabel("behavior: --")
        self.lbl_status.setStyleSheet("color:#06c;")
        rv.addWidget(self.lbl_status)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(300)
        rv.addWidget(self.log, 1)
        split.addWidget(right)

        split.setSizes([220, 700, 340])

    # ────────────────────────────────────────────────────────────
    # 헬퍼
    # ────────────────────────────────────────────────────────────
    def _log(self, s: str):
        self.log.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {s}")

    def _is_kick(self) -> bool:
        return "kick" in os.path.basename(self.doc.path).lower()

    def _slider_values(self) -> dict:
        return {n: self.sliders[n].value() / SLIDER_SCALE for n in JOINT_ORDER}

    def _set_sliders(self, joints: dict):
        """슬라이더에 값을 채운다. 모터 한계 밖 값은 명시적으로 알린다.

        ★ 조용히 clamp 하면 안 된다.
          기존 YAML 에 한계 밖 값이 들어 있으면(모터가 거부해 실제로는 그
          자세가 안 나온다) 사용자는 원인을 영영 모른다. QSlider 는 범위 밖
          setValue 를 자동으로 자르므로, 자르기 '전에' 비교해서 로그와 라벨로
          드러낸다.
        """
        self._loading = True
        out = []
        try:
            for n in JOINT_ORDER:
                v = float(joints.get(n, 0.0))
                lim = self._limits.get(n)
                clipped = False
                if lim:
                    lo, hi = lim
                    if v < lo - 1e-9 or v > hi + 1e-9:
                        clipped = True
                        out.append((n, v, lo, hi))
                self.sliders[n].setValue(int(round(v * SLIDER_SCALE)))
                shown = self.sliders[n].value() / SLIDER_SCALE
                self.val_labels[n].setText(f"{shown:+.3f}")
                self.val_labels[n].setStyleSheet(
                    "color:#c00; font-weight:bold;" if clipped else "")
                self.lim_labels[n].setStyleSheet(
                    "color:#c00;" if clipped else "color:#888;")
        finally:
            self._loading = False

        for n, v, lo, hi in out:
            shown = self.sliders[n].value() / SLIDER_SCALE
            self._log(f"★ 한계 초과: {n} = {v:+.3f} rad "
                      f"(허용 {lo:+.3f}~{hi:+.3f}) → {shown:+.3f} 로 표시. "
                      f"모터가 거부해 실제로는 적용되지 않던 값입니다.")

    def _retry_limits(self):
        """한계 응답이 올 때까지 2초 간격으로 최대 10회 재시도."""
        if self._limits:
            self.limit_timer.stop()
            return
        self._limit_retry += 1
        if self._limit_retry > 10:
            self.limit_timer.stop()
            self._log("관절 한계를 못 받았습니다 — 슬라이더는 ±"
                      f"{JOINT_RANGE_FALLBACK} rad 기본 범위로 동작합니다. "
                      "한계 밖 값은 모터가 조용히 거부하니 주의하세요.")
            return
        self.ros.request_limits()

    def _apply_limits(self, limits: dict):
        """받은 한계를 슬라이더 범위와 limit 라벨에 반영한다."""
        self._limits = limits
        for n in JOINT_ORDER:
            lim = limits.get(n)
            if not lim:
                self.lim_labels[n].setText("(미수신)")
                continue
            lo, hi = lim
            # 슬라이더 범위를 실제 한계로 좁힌다. 이 밖은 애초에 선택 불가.
            self.sliders[n].setRange(int(round(lo * SLIDER_SCALE)),
                                     int(round(hi * SLIDER_SCALE)))
            self.sliders[n].setToolTip(
                f"{n}\nEEPROM 한계: {lo:+.3f} ~ {hi:+.3f} rad")
            self.lim_labels[n].setText(f"{lo:+.2f}~{hi:+.2f}")
        # 범위가 바뀌면 현재 phase 값이 잘렸을 수 있으므로 다시 그린다.
        if self.cur_idx >= 0:
            self._on_phase_selected(self.cur_idx)

    def _tick(self):
        # reload 응답 뒤 이어서 실행할 동작 (ROS 스레드에서 큐잉된 것)
        if self._after_reload is not None:
            fn = self._after_reload
            self._after_reload = None
            fn()

        limits, msg, is_new = self.ros.get_limits()
        if is_new:
            self._log(msg)
            if limits:
                self._apply_limits(limits)

        pos = self.ros.get_positions()
        for n in JOINT_ORDER:
            if n in pos:
                self.cur_labels[n].setText(f"{pos[n]:+.3f}")
        b = self.ros.get_behavior()
        if b:
            parts = []
            for k in ("standup", "kick", "robot_init"):
                sub = b.get(k)
                if isinstance(sub, dict) and sub.get("active"):
                    parts.append(f"{k}: {sub.get('phase', '?')}")
            self.lbl_status.setText(
                "behavior: " + (", ".join(parts) if parts else "idle"))

        # 속도 기입 후 한 사이클 뒤에 목표를 보낸다(순서 보장).
        if self._pending_apply is not None:
            goals, t_due = self._pending_apply
            if time.time() >= t_due:
                self._pending_apply = None
                self.ros.publish_goals(goals)
                self._log(f"Apply 전송: {len(goals)} joints")

    # ────────────────────────────────────────────────────────────
    # 파일 / 목록
    # ────────────────────────────────────────────────────────────
    def _on_open(self):
        from PySide6.QtWidgets import QFileDialog
        start = os.path.dirname(self.doc.path) or default_yaml_path() or "."
        path, _ = QFileDialog.getOpenFileName(
            self, "키프레임 YAML 열기", start, "YAML (*.yaml *.yml)")
        if path:
            self._load_path(path)

    def _load_path(self, path: str):
        try:
            self.doc.load(path)
        except Exception as e:
            QMessageBox.critical(self, "로드 실패", f"{path}\n\n{e}")
            return
        info = diagnose_save_path(path)
        tag = ""
        if info["is_link"]:
            tag = "  [symlink → 소스와 동일 파일]"
        elif info["risky"]:
            tag = "  [★ install 복사본 — 빌드 시 덮어써짐]"
        self.lbl_path.setText(path + tag)
        self.lbl_path.setToolTip(f"열린 경로: {path}\n실제 파일: {info['real']}")
        self.lbl_path.setStyleSheet(
            "color:#c00; font-weight:bold;" if info["risky"] else "color:#555;")
        if info["risky"]:
            self._log("★ 이 파일은 install 트리의 복사본입니다. "
                      "다음 colcon build 때 편집 내용이 덮여 사라집니다. "
                      "--symlink-install 로 빌드하는 것을 권합니다.")
        self._robot_stale = False
        self._update_stale_label()
        self.cmb_seq.blockSignals(True)
        self.cmb_seq.clear()
        self.cmb_seq.addItems(self.doc.seq_keys)
        self.cmb_seq.blockSignals(False)
        self.cur_seq = self.doc.seq_keys[0]
        self.cmb_seq.setCurrentText(self.cur_seq)
        self._refresh_list(select=0)
        self._log(f"로드: {path}  시퀀스 {self.doc.seq_keys}")

    def _on_seq_changed(self, key: str):
        if not key:
            return
        self.cur_seq = key
        self._refresh_list(select=0)

    def _refresh_list(self, select: int = -1):
        self.lst.blockSignals(True)
        self.lst.clear()
        for i, n in enumerate(self.doc.phase_names(self.cur_seq)):
            self.lst.addItem(f"{i}: {n}")
        self.lst.blockSignals(False)
        if self.lst.count():
            self.lst.setCurrentRow(max(0, min(select, self.lst.count() - 1)))

    def _on_phase_selected(self, row: int):
        if row < 0 or row >= len(self.doc.phases(self.cur_seq)):
            self.cur_idx = -1
            return
        self.cur_idx = row
        ph = self.doc.get_phase(self.cur_seq, row)
        self.spn_rpm.setValue(float(ph.get("rpm", 30.0)))
        self.spn_hold.setValue(float(ph.get("hold_sec", 0.5)))
        self._set_sliders(ph.get("joints", {}))

    # ────────────────────────────────────────────────────────────
    # phase 편집
    # ────────────────────────────────────────────────────────────
    def _current_phase_dict(self) -> dict:
        return {
            "name": self.doc.phase_names(self.cur_seq)[self.cur_idx],
            "rpm": float(self.spn_rpm.value()),
            "hold_sec": float(self.spn_hold.value()),
            "joints": self._slider_values(),
        }

    def _on_slider(self):
        if self._loading:
            return
        for n in JOINT_ORDER:
            self.val_labels[n].setText(
                f"{self.sliders[n].value() / SLIDER_SCALE:+.3f}")

    def _on_sync(self):
        pos = self.ros.get_positions()
        missing = [n for n in JOINT_ORDER if n not in pos]
        if missing:
            self._log(f"경고: /joint_states 에 없는 관절 {len(missing)}개 "
                      f"→ 0 으로 둠 ({missing[:3]}...)")
        self._set_sliders(pos)
        self._log("Sync: 실제 위치를 슬라이더로 가져옴")

    def _on_mirror(self, r_to_l: bool):
        vals = self._slider_values()
        for r, ln in MIRROR_PAIRS:
            if r_to_l:
                vals[ln] = -vals[r]
            else:
                vals[r] = -vals[ln]
        self._set_sliders(vals)
        self._log(f"Mirror {'R→L' if r_to_l else 'L→R'} 적용 (부호 반전)")

    def _on_save_phase(self):
        if self.cur_idx < 0:
            return
        self.doc.set_phase(self.cur_seq, self.cur_idx,
                           self._current_phase_dict())
        self._robot_stale = True
        self._update_stale_label()
        self._log(f"phase '{self._current_phase_dict()['name']}' 버퍼에 저장. "
                  f"★ 로봇에는 아직 반영 안 됨 — Play/Goto 시 자동으로 "
                  f"저장+reload 합니다.")

    def _on_revert_phase(self):
        self._on_phase_selected(self.cur_idx)
        self._log("phase 를 마지막 저장 상태로 되돌림")

    def _on_add(self):
        name, ok = QInputDialog.getText(self, "Add Phase", "phase 이름:")
        if not ok or not name.strip():
            return
        idx = self.cur_idx + 1 if self.cur_idx >= 0 else 0
        self.doc.insert_phase(self.cur_seq, idx, {
            "name": name.strip(), "rpm": 30.0, "hold_sec": 0.5,
            "joints": {n: 0.0 for n in JOINT_ORDER},
        })
        self._refresh_list(select=idx)
        self._log(f"phase '{name}' 추가 (index {idx})")

    def _on_dup(self):
        if self.cur_idx < 0:
            return
        name, ok = QInputDialog.getText(
            self, "Duplicate Phase", "새 phase 이름:",
            text=self.doc.phase_names(self.cur_seq)[self.cur_idx] + "_copy")
        if not ok or not name.strip():
            return
        src = copy.deepcopy(self.doc.get_phase(self.cur_seq, self.cur_idx))
        src["name"] = name.strip()
        idx = self.cur_idx + 1
        self.doc.insert_phase(self.cur_seq, idx, src)
        self._refresh_list(select=idx)
        self._log(f"phase 복제 → '{name}' (index {idx})")

    def _on_del(self):
        if self.cur_idx < 0:
            return
        n = self.doc.phase_names(self.cur_seq)[self.cur_idx]
        if QMessageBox.question(
                self, "Delete Phase",
                f"'{n}' 을 삭제할까요?") != QMessageBox.Yes:
            return
        idx = self.cur_idx
        self.doc.remove_phase(self.cur_seq, idx)
        self._refresh_list(select=max(0, idx - 1))
        self._log(f"phase '{n}' 삭제")

    def _on_rename(self):
        if self.cur_idx < 0:
            return
        old = self.doc.phase_names(self.cur_seq)[self.cur_idx]
        name, ok = QInputDialog.getText(
            self, "Rename Phase", "새 이름:", text=old)
        if not ok or not name.strip():
            return
        self.doc.get_phase(self.cur_seq, self.cur_idx)["name"] = name.strip()
        self.doc.dirty = True
        self._refresh_list(select=self.cur_idx)
        self._log(f"phase 이름 '{old}' → '{name}'")

    def _on_move(self, delta: int):
        if self.cur_idx < 0:
            return
        new = self.doc.move_phase(self.cur_seq, self.cur_idx, delta)
        self._refresh_list(select=new)
        self._log(f"phase 순서 이동 → index {new}")

    # ────────────────────────────────────────────────────────────
    # 로봇 동작
    # ────────────────────────────────────────────────────────────
    def _on_apply(self):
        goals = self._slider_values()
        rpm = float(self.spn_rpm.value())
        # 속도를 먼저 기입하고, 한 제어 사이클(20ms)보다 넉넉히 뒤에 목표를 보낸다.
        # 붙여서 쏘면 목표가 먼저 버스로 나가 직전 속도로 출발할 수 있다.
        self.ros.publish_speed(rpm)
        self._pending_apply = (goals, time.time() + 0.08)
        self._log(f"Apply: rpm={rpm:.0f} 기입 → 80ms 후 목표 전송")

    def _set_full_pwm(self, on: bool, quiet: bool = True):
        """팔 Goal PWM 100% 모드를 켜고 끈다."""
        self._full_pwm = bool(on)
        if self._full_pwm:
            # 0.0 = "그 모터의 PWM Limit(=100%)" 이라고 노드가 해석한다.
            self.ros.publish_goal_pwm(FULL_PWM_JOINTS, 0.0)
            self.btn_pwm.setText("Arms PWM: 100% (편집용)")
            self.btn_pwm.setStyleSheet("background:#2a6; color:white;")
            if not quiet:
                self._log("팔 Goal PWM 을 100% 로 승격 "
                          "(편집 중 유지, 1Hz 재기입)")
        else:
            self.btn_pwm.setText("Arms PWM: YAML 운용값")
            self.btn_pwm.setStyleSheet("")
            self.ros.call_reload_gains(
                lambda ok, m: self._log(f"PWM 원복(reload_gains): {ok} {m}"))
            if not quiet:
                self._log("팔 Goal PWM 을 motor_gains.yaml 운용값으로 원복")

    def _on_toggle_pwm(self):
        self._set_full_pwm(not self._full_pwm, quiet=False)

    def _reassert_full_pwm(self):
        """1Hz. behavior 가 종료하며 되돌린 PWM 을 다시 100% 로 덮어쓴다."""
        if self._full_pwm:
            self.ros.publish_goal_pwm(FULL_PWM_JOINTS, 0.0)

    def _on_torque(self, enable: bool):
        if not enable:
            r = QMessageBox.question(
                self, "Torque OFF",
                "전신 토크를 끕니다. 로봇이 주저앉을 수 있습니다.\n"
                "누워 있는 상태인지 확인하셨나요?")
            if r != QMessageBox.Yes:
                return
        self.ros.call_torque(
            enable, lambda ok, m: self._log(f"torque {enable}: {ok} {m}"))

    def _seq_direction(self) -> str:
        """현재 시퀀스 키에서 standup 의 fall_direction 인자를 뽑는다."""
        return "backward" if "backward" in self.cur_seq else "forward"

    def _seq_leg(self) -> str:
        """현재 시퀀스 키에서 kick 의 leg 인자를 뽑는다.

        kick_keyframes.yaml 은 right_kick_sequence / left_kick_sequence 구조라
        키 이름만 보면 어느 다리인지 알 수 있다.
        """
        return "left" if self.cur_seq.startswith("left") else "right"

    def _update_stale_label(self):
        if self._robot_stale:
            self.lbl_stale.setText("로봇 반영: 필요 (편집됨)")
            self.lbl_stale.setStyleSheet("color:#c60; font-weight:bold;")
        else:
            self.lbl_stale.setText("로봇 반영: 최신")
            self.lbl_stale.setStyleSheet("color:#2a6;")

    def _sync_then(self, action, label: str):
        """편집 내용을 로봇에 반영한 뒤 action 을 실행한다.

        Save Phase 는 메모리 버퍼만 바꾸므로, 재생 계열 명령 전에
        (필요하면) 파일 저장 → reload → 그 다음에 실행해야 한다.
        이 순서를 사용자가 외우게 하지 않는 것이 이 함수의 목적이다.
        """
        if not self._robot_stale:
            action()
            return
        # 1) 현재 슬라이더 상태까지 버퍼에 반영하고 파일로 저장
        if not self._save_yaml_internal():
            self._log(f"{label} 취소: 저장 실패")
            return
        # 2) reload → 3) 응답이 오면 action (GUI 스레드에서 _tick 이 실행)
        self._log(f"{label} 전에 자동 저장 + reload 진행")

        def _done(ok, msg):
            def _next():
                self._log(f"reload: {ok} {msg}")
                if ok:
                    self._robot_stale = False
                    self._update_stale_label()
                    action()
                else:
                    self._log(f"{label} 취소: reload 실패")
            self._after_reload = _next

        if self._is_kick():
            self.ros.call_kick("reload", leg=self._seq_leg(), done_cb=_done)
        else:
            self.ros.call_standup("reload",
                                  fall_direction=self._seq_direction(),
                                  done_cb=_done)

    def _on_goto(self):
        if self.cur_idx < 0:
            return
        name = self.doc.phase_names(self.cur_seq)[self.cur_idx]

        def _run():
            cb = lambda ok, m: self._log(f"goto {name}: {ok} {m}")  # noqa: E731
            if self._is_kick():
                self.ros.call_kick(f"goto {name}", leg=self._seq_leg(),
                                   done_cb=cb)
            else:
                self.ros.call_standup(f"goto {name}",
                                      fall_direction=self._seq_direction(),
                                      done_cb=cb)

        self._sync_then(_run, f"goto {name}")

    def _on_reload(self):
        def _done(ok, msg):
            def _next():
                self._log(f"reload: {ok} {msg}")
                if ok:
                    self._robot_stale = False
                    self._update_stale_label()
            self._after_reload = _next
        if self._is_kick():
            self.ros.call_kick("reload", leg=self._seq_leg(), done_cb=_done)
        else:
            self.ros.call_standup("reload",
                                  fall_direction=self._seq_direction(),
                                  done_cb=_done)

    def _on_seq_cmd(self, cmd: str):
        cb = lambda ok, m: self._log(f"{cmd}: {ok} {m}")  # noqa: E731
        if self._is_kick():
            self.ros.call_kick(cmd, leg=self._seq_leg(), done_cb=cb)
        else:
            self.ros.call_standup(cmd, fall_direction=self._seq_direction(),
                                  done_cb=cb)

    def _on_play(self):
        r = QMessageBox.question(
            self, "Play Sequence",
            f"'{self.cur_seq}' 전체를 자동 재생합니다.\n"
            "로봇 주변에 사람/장애물이 없는지 확인하셨나요?")
        if r != QMessageBox.Yes:
            return
        self._sync_then(lambda: self._on_seq_cmd("start"), "play")

    def _save_yaml_internal(self) -> bool:
        """현재 편집 상태를 파일로 저장한다. 성공 여부를 반환."""
        if not self.doc.path:
            self._log("저장할 파일이 없습니다 ([Open YAML])")
            return False
        if self.cur_idx >= 0:
            self.doc.set_phase(self.cur_seq, self.cur_idx,
                               self._current_phase_dict())
        try:
            path = self.doc.save()
        except Exception as e:
            QMessageBox.critical(self, "저장 실패", str(e))
            return False

        info = diagnose_save_path(path)
        self._log(f"저장: {path}")
        if info["real"] != path:
            self._log(f"  실제 파일: {info['real']}"
                      f"{'  (심볼릭 링크 → 소스와 동일 파일)' if info['is_link'] else ''}")
        self._log(f"  백업: {os.path.basename(path)}.bak")

        if info["risky"]:
            # ★ 여기가 데이터 유실 지점이다. 복사본에 저장하면 다음 빌드에 날아간다.
            msg = (f"{path}\n\n"
                   "이 파일은 install 트리의 '복사본' 입니다 "
                   "(심볼릭 링크가 아님).\n"
                   "다음 colcon build 때 소스 파일로 덮여 편집 내용이 사라집니다.\n\n")
            if info["src_guess"]:
                msg += f"소스로 보이는 경로:\n{info['src_guess']}\n\n여기에도 저장할까요?"
                r = QMessageBox.warning(
                    self, "빌드 시 덮어써질 수 있음", msg,
                    QMessageBox.Yes | QMessageBox.No)
                if r == QMessageBox.Yes:
                    try:
                        shutil.copy2(path, info["src_guess"])
                        self._log(f"  소스에도 저장: {info['src_guess']}")
                    except Exception as e:
                        self._log(f"  ★ 소스 저장 실패: {e}")
            else:
                msg += ("소스 경로를 찾지 못했습니다. "
                        "--symlink-install 로 다시 빌드하거나, "
                        "소스 트리의 config/ 를 직접 열어 편집하세요.")
                QMessageBox.warning(self, "빌드 시 덮어써질 수 있음", msg)
        return True

    def _on_save_yaml(self):
        if self._save_yaml_internal():
            self._robot_stale = True     # 파일만 바뀌었을 뿐 노드는 아직 구버전
            self._update_stale_label()
            self._log("→ [Reload on Robot] 을 눌러야 노드가 새 YAML 을 읽습니다. "
                      "(Play/Goto 는 자동으로 처리합니다)")

    def closeEvent(self, event):
        if self.doc.dirty:
            r = QMessageBox.question(
                self, "종료", "저장하지 않은 편집이 있습니다. 그래도 닫을까요?")
            if r != QMessageBox.Yes:
                event.ignore()
                return
        # 편집용으로 올려둔 팔 PWM 상한을 반드시 원복한다.
        # 안 내리면 100% 인 채로 보행에 들어가 낙상 시 감속기 보호가 사라진다.
        if self._full_pwm:
            self._set_full_pwm(False, quiet=False)
            # 서비스 호출이 나갈 시간을 준다(spin 스레드가 아직 살아 있음).
            time.sleep(0.3)
        try:
            self.timer.stop()
            self.pwm_timer.stop()
            self.limit_timer.stop()
            self.ros.destroy_node()
        except Exception:
            pass
        super().closeEvent(event)


def main():
    rclpy.init()
    ros = RosNode()
    spin = RosSpinThread(ros)
    spin.start()

    app = QApplication(sys.argv)
    win = MainWindow(ros)
    win.show()
    try:
        code = app.exec()
    finally:
        spin.stop()
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass
    sys.exit(code)


if __name__ == "__main__":
    main()
