#!/usr/bin/env python3
"""
pose_slider_gui.py
==================

전신 18개 조인트를 슬라이더로 조정해서 자세를 잡고, standup_keyframes.yaml
형식의 phase 블록으로 출력해주는 PySide6 기반 GUI.

워크플로:
  1) 슬라이더로 원하는 자세 만든다.
  2) Apply 버튼 → 모터에 전송 (안전상 명시적으로 누름).
  3) 자세가 마음에 들면 phase name 입력 후 "Copy as YAML" → 클립보드에 복사
     → standup_keyframes.yaml 에 붙여넣기.

추가 기능:
  - Torque ON/OFF: 토크 끄고 손으로 로봇 자세를 직접 잡은 후 (kinesthetic
    teaching) "Sync from Robot" 버튼으로 슬라이더에 그 자세 가져오기.
  - Sync from Robot: 현재 /joint_states 의 실제 위치로 모든 슬라이더 초기화.
  - Profile RPM: Apply 시 적용될 모터 속도. 너무 빠르면 위험하니 기본 20.

연결:
  발행: /joint_goals (sensor_msgs/JointState)  Apply 시
  구독: /joint_states (sensor_msgs/JointState) 실제 위치 표시 + Sync 용
  서비스: /u2d2/torque_enable                  토크 토글

실행:
  ros2 run humanoid_serial_bridge pose_slider_gui
  (X11 forwarding 또는 NoMachine 등 GUI 환경 필요)

의존성:
  PySide6 가 설치돼 있어야 함:
    pip3 install PySide6
"""
import sys
import threading
from collections import OrderedDict

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from humanoid_behavior_interfaces.srv import (
    TorqueEnable, TorqueEnableJoints, GetJointLimits,
)

# PySide6 import 는 import 실패 시 친절한 에러 메시지를 위해 try 로 감쌈.
# (사용자가 PySide6 설치 안 한 경우 흔함)
try:
    from PySide6.QtCore import Qt, QTimer, Signal, QObject
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QGridLayout, QGroupBox, QLabel, QSlider, QPushButton, QLineEdit,
        QTextEdit, QDoubleSpinBox, QSpinBox, QSplitter, QMessageBox,
        QCheckBox,
    )
except ImportError as e:
    print("ERROR: PySide6 가 설치돼 있지 않습니다.")
    print("       pip3 install PySide6")
    print(f"       원인: {e}")
    sys.exit(1)


# ────────────────────────────────────────────────────────────────────
# 조인트 정의
# ────────────────────────────────────────────────────────────────────
# (joint_name, min_rad, max_rad, default_rad) — robot_init_behavior 의 init pose 값을 default 로.
# min/max 는 일단 ±π 로 넓게 잡고 사용자가 필요시 수정. URDF 가 있으면 거기서
# 자동으로 가져오는 게 정확하지만 KHUBER 는 URDF 가 직접 코드/yaml 에 흩어져
# 있어 일단 보수적인 디폴트 사용.
JOINT_RANGE_DEFAULT = 3.14   # ±π
LEG_JOINTS = [
    # 오른쪽 다리 (init pose: pitch=0.8, knee=1.4, ankle_pitch=0.7)
    ("r_pelvis_pitch_joint",  0.8),
    ("r_pelvis_roll_joint",   0.0),
    ("r_pelvis_yaw_joint",    0.0),
    ("r_knee_pitch_joint",    1.4),
    ("r_ankle_pitch_joint",   0.7),
    ("r_ankle_roll_joint",    0.0),
    # 왼쪽 다리 (대칭이라 부호 반전)
    ("l_pelvis_pitch_joint", -0.8),
    ("l_pelvis_roll_joint",   0.0),
    ("l_pelvis_yaw_joint",    0.0),
    ("l_knee_pitch_joint",   -1.4),
    ("l_ankle_pitch_joint",  -0.7),
    ("l_ankle_roll_joint",    0.0),
]
ARM_JOINTS = [
    # 오른쪽 팔 (init pose: shoulder_pitch=0.2, elbow=2.967)
    ("r_shoulder_pitch_joint", 0.2),
    ("r_shoulder_roll_joint",  0.0),
    ("r_elbow_pitch_joint",    2.967),
    # 왼쪽 팔
    ("l_shoulder_pitch_joint", -0.2),
    ("l_shoulder_roll_joint",   0.0),
    ("l_elbow_pitch_joint",    -2.967),
]
ALL_JOINTS = LEG_JOINTS + ARM_JOINTS

# 슬라이더 해상도: QSlider 는 int 만 지원하므로 rad 값을 1000 배해서 사용.
# 즉 슬라이더 한 칸 = 0.001 rad ≈ 0.057°. 충분히 정밀.
SLIDER_SCALE = 1000


# ────────────────────────────────────────────────────────────────────
# ROS 노드 — GUI 와 분리해서 별도 스레드에서 spin
# ────────────────────────────────────────────────────────────────────
# GUI 메인 스레드와 ROS executor 스레드를 분리하는 게 핵심.
# - GUI 가 ROS callback 안에서 직접 위젯 업데이트하면 Qt thread-safety 위반.
# - 그래서 콜백은 받은 데이터를 thread-safe 큐 같은 곳에 두고, Qt timer 가
#   주기적으로 읽어서 위젯 업데이트.
# - 여기서는 단순히 lock 으로 보호되는 dict 에 저장해 두고 100ms Qt timer 로 갱신.
class RosNode(Node):
    def __init__(self):
        super().__init__('pose_slider_gui')

        # 발행/구독/클라이언트
        self.pub_goals = self.create_publisher(JointState, '/joint_goals', 10)
        self.sub_states = self.create_subscription(
            JointState, '/joint_states', self._on_joint_states, 10)
        self.cli_torque = self.create_client(TorqueEnable, '/u2d2/torque_enable')
        # 특정 조인트만 토크 ON/OFF 하는 클라이언트.
        # body_motor_controller_node 의 cb_torque_enable_joints 가 처리.
        self.cli_torque_joints = self.create_client(
            TorqueEnableJoints, '/u2d2/torque_enable_joints')
        # 모든 조인트의 EEPROM Min/Max Position Limit 조회 클라이언트.
        # GUI 시작 시 한번 호출해서 슬라이더 범위를 모터 실제 한계에 맞춤.
        self.cli_get_limits = self.create_client(
            GetJointLimits, '/u2d2/get_joint_limits')

        # 최신 joint_states 캐시. dict[name] = position(rad).
        # GUI 스레드에서 읽으니까 lock 필수.
        self._js_lock = threading.Lock()
        self._latest_positions = {}

        self.get_logger().info("pose_slider_gui ROS node ready")

    def _on_joint_states(self, msg: JointState):
        """/joint_states 콜백. 단순히 캐시에 저장만."""
        with self._js_lock:
            for n, p in zip(msg.name, msg.position):
                self._latest_positions[n] = float(p)

    def get_positions(self) -> dict:
        """GUI 가 호출. 현재 캐시된 실제 위치의 스냅샷 반환."""
        with self._js_lock:
            return dict(self._latest_positions)

    def publish_goals(self, joint_dict: dict):
        """
        joint_dict = {name: rad} 를 /joint_goals 로 발행.
        body_motor_controller_node 의 cb_goals 가 받아서 다음 cycle 에 SyncWrite.
        """
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(joint_dict.keys())
        msg.position = [float(v) for v in joint_dict.values()]
        self.pub_goals.publish(msg)

    def call_torque(self, enable: bool, done_cb=None):
        """
        토크 ON/OFF 서비스 호출. 비동기 (GUI 가 멈추면 안 되므로).
        done_cb(success: bool, message: str) 콜백이 ROS 스레드에서 호출됨.
        """
        if not self.cli_torque.service_is_ready():
            if done_cb:
                done_cb(False, "/u2d2/torque_enable 서비스 준비 안 됨")
            return
        req = TorqueEnable.Request()
        req.enable = bool(enable)
        future = self.cli_torque.call_async(req)
        if done_cb:
            def _cb(fut):
                try:
                    res = fut.result()
                    done_cb(bool(res.success), str(res.message))
                except Exception as e:
                    done_cb(False, f"exception: {e}")
            future.add_done_callback(_cb)

    def call_torque_joints(self, joint_names: list, enable: bool, done_cb=None):
        """
        특정 조인트들의 토크 ON/OFF.
        joint_names: list[str] — 변경할 조인트 이름들
        enable: True=ON, False=OFF
        done_cb(success: bool, message: str, applied: list[str]) — 응답 콜백
        """
        if not self.cli_torque_joints.service_is_ready():
            if done_cb:
                done_cb(False, "/u2d2/torque_enable_joints 서비스 준비 안 됨", [])
            return
        req = TorqueEnableJoints.Request()
        req.joint_names = list(joint_names)
        req.enable = bool(enable)
        future = self.cli_torque_joints.call_async(req)
        if done_cb:
            def _cb(fut):
                try:
                    res = fut.result()
                    done_cb(bool(res.success), str(res.message), list(res.applied_joints))
                except Exception as e:
                    done_cb(False, f"exception: {e}", [])
            future.add_done_callback(_cb)

    def call_get_joint_limits(self, done_cb=None, timeout_sec: float = 3.0):
        """
        모든 조인트의 EEPROM 위치 한계 조회.
        done_cb(success: bool, message: str, limits: dict[name -> (lower, upper)])
        timeout_sec: 서비스 ready 대기 시간. GUI 시작 직후 호출하면 노드 아직
                     안 떠 있을 수 있으니 짧게 대기.
        """
        if not self.cli_get_limits.wait_for_service(timeout_sec=timeout_sec):
            if done_cb:
                done_cb(False, "/u2d2/get_joint_limits 서비스 timeout", {})
            return
        req = GetJointLimits.Request()
        future = self.cli_get_limits.call_async(req)
        if done_cb:
            def _cb(fut):
                try:
                    res = fut.result()
                    # 응답의 병렬 배열을 dict 로 변환해서 GUI 사용 편하게.
                    limits = {}
                    for n, lo, up in zip(
                        res.joint_names, res.lower_limits_rad, res.upper_limits_rad
                    ):
                        limits[n] = (float(lo), float(up))
                    done_cb(bool(res.success), str(res.message), limits)
                except Exception as e:
                    done_cb(False, f"exception: {e}", {})
            future.add_done_callback(_cb)


class RosSpinThread(threading.Thread):
    """
    ROS executor 를 별도 스레드에서 spin. GUI 메인 루프와 분리.
    daemon=True 라 GUI 종료 시 자동 정리.
    """
    def __init__(self, node: Node):
        super().__init__(daemon=True)
        self.node = node
        self._stop = threading.Event()

    def run(self):
        # spin_until_future_complete 가 아니라 spin 으로 계속 돌림.
        # _stop 플래그는 사실 daemon=True 면 안 써도 되지만, 명시적 종료 시
        # 깔끔하게 끊기 위해 함께 둠.
        try:
            while not self._stop.is_set() and rclpy.ok():
                rclpy.spin_once(self.node, timeout_sec=0.1)
        except Exception as e:
            self.node.get_logger().error(f"spin thread 종료 (예외): {e}")

    def stop(self):
        self._stop.set()


# ────────────────────────────────────────────────────────────────────
# 단일 조인트 슬라이더 위젯
# ────────────────────────────────────────────────────────────────────
class JointSliderRow(QWidget):
    """
    한 조인트당 한 줄:
      [토크CK] [조인트 이름]  [슬라이더]  [슬라이더값(rad)]  [실제 위치(rad)]  [Sync 버튼]

    체크박스 동작:
      - 체크됨 = 토크 ON 으로 보고 싶음
      - 체크 해제 = 토크 OFF 로 만들고 손으로 잡고 싶음
      - 사용자가 체크박스를 클릭하면 torque_toggle_requested 시그널이 발화.
        MainWindow 가 받아서 ROS 서비스 호출. 응답이 오면 set_torque_state()
        로 체크박스 상태를 실제 결과에 맞춰 갱신 (요청 실패 시 원복).
      - "프로그램이 상태를 강제로 바꾸는" 경우(예: Sync from torque state)에는
        체크박스가 시그널을 발화하지 않도록 setChecked 호출 시 blockSignals 사용.
    """
    # 시그널 정의: (joint_name, want_enable) — 사용자가 토크 토글을 요청
    torque_toggle_requested = Signal(str, bool)

    def __init__(self, joint_name: str, default_rad: float,
                 min_rad: float = -JOINT_RANGE_DEFAULT,
                 max_rad: float = JOINT_RANGE_DEFAULT):
        super().__init__()
        self.joint_name = joint_name
        self.min_rad = min_rad
        self.max_rad = max_rad

        layout = QHBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        self.setLayout(layout)

        # 0) 토크 체크박스 — 체크=ON, 해제=OFF.
        # 초기 상태는 "알 수 없음" 이라 일단 체크된 상태로 표시 (대부분 ON 으로 시작).
        # 실제 모터 상태를 알기 어려우니 사용자가 직접 토글하면 그 결과로 동기화.
        self.chk_torque = QCheckBox()
        self.chk_torque.setChecked(True)
        self.chk_torque.setToolTip(
            "이 관절의 토크 ON/OFF.\n"
            "해제하면 손으로 자세를 잡을 수 있음 (kinesthetic teaching).\n"
            "주의: 자세 무너짐 방지를 위해 받침대 사용 권장."
        )
        # toggled (사용자 클릭) → 외부로 시그널 발화
        self.chk_torque.toggled.connect(self._on_torque_checkbox_toggled)
        layout.addWidget(self.chk_torque)

        # 1) 조인트 이름 — 너비 고정해서 정렬 깔끔
        self.lbl_name = QLabel(joint_name)
        self.lbl_name.setMinimumWidth(180)
        self.lbl_name.setMaximumWidth(220)
        layout.addWidget(self.lbl_name)

        # 2) 슬라이더 — 가로로 늘어남
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(int(min_rad * SLIDER_SCALE))
        self.slider.setMaximum(int(max_rad * SLIDER_SCALE))
        self.slider.setValue(int(default_rad * SLIDER_SCALE))
        # 100 단위 마크 (= 0.1 rad ≈ 5.7°)
        self.slider.setTickInterval(int(0.5 * SLIDER_SCALE))
        self.slider.setTickPosition(QSlider.TicksBelow)
        layout.addWidget(self.slider, stretch=1)

        # 3) 현재 슬라이더 값 (rad). 슬라이더 변경 시 갱신.
        self.lbl_value = QLabel(f"{default_rad:+.3f}")
        self.lbl_value.setMinimumWidth(70)
        self.lbl_value.setMaximumWidth(70)
        self.lbl_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self.lbl_value)

        # 4) 실제 모터 위치. 회색으로 표시 (참고용)
        self.lbl_actual = QLabel("---")
        self.lbl_actual.setMinimumWidth(70)
        self.lbl_actual.setMaximumWidth(70)
        self.lbl_actual.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_actual.setStyleSheet("color: gray;")
        layout.addWidget(self.lbl_actual)

        # 5) 이 조인트만 실제 값으로 동기화
        self.btn_sync = QPushButton("←")
        self.btn_sync.setMaximumWidth(28)
        self.btn_sync.setToolTip("이 슬라이더를 실제 모터 위치로 동기화")
        layout.addWidget(self.btn_sync)

        # 시그널 연결
        self.slider.valueChanged.connect(self._on_slider_changed)
        # Sync 버튼은 외부에서 callback 연결 (RosNode 접근 필요)

    def _on_torque_checkbox_toggled(self, checked: bool):
        """사용자가 체크박스 토글 → 외부로 시그널 전달."""
        self.torque_toggle_requested.emit(self.joint_name, bool(checked))

    def set_torque_state(self, enabled: bool):
        """
        프로그램이 토크 상태를 강제로 반영시키는 경우 (예: 서비스 응답이
        실패해서 원복하거나, 외부에서 가져온 상태를 표시). 시그널 발화를
        막기 위해 blockSignals 사용.
        """
        self.chk_torque.blockSignals(True)
        self.chk_torque.setChecked(bool(enabled))
        self.chk_torque.blockSignals(False)

    def _on_slider_changed(self, val_int: int):
        rad = val_int / SLIDER_SCALE
        self.lbl_value.setText(f"{rad:+.3f}")

    def get_value(self) -> float:
        """현재 슬라이더의 rad 값."""
        return self.slider.value() / SLIDER_SCALE

    def set_value(self, rad: float):
        """슬라이더를 주어진 rad 값으로 설정 (범위 clamp)."""
        rad = max(self.min_rad, min(self.max_rad, rad))
        self.slider.setValue(int(rad * SLIDER_SCALE))

    def update_actual(self, rad: float | None):
        """실제 모터 위치 표시 라벨 갱신. None 이면 '---'."""
        if rad is None:
            self.lbl_actual.setText("---")
        else:
            self.lbl_actual.setText(f"{rad:+.3f}")

    def set_limits(self, min_rad: float, max_rad: float):
        """
        슬라이더 범위를 변경. 모터 EEPROM 한계값으로 동적 설정할 때 사용.

        주의:
          - 현재 슬라이더 값이 새 범위 밖이면 자동으로 clamp 됨 (QSlider 기본 동작).
            그 결과 시그널이 발화되어 lbl_value 도 자동 갱신됨.
          - min == max 같은 비정상 값이 들어오면 무시 (모터 한계 설정이 잘못된 경우).
          - tooltip 도 새 범위 반영해서 갱신.
        """
        if min_rad >= max_rad:
            # 비정상 — 기본값 유지
            return
        self.min_rad = float(min_rad)
        self.max_rad = float(max_rad)
        # QSlider 범위 변경 시 현재 값이 밖이면 자동 clamp + valueChanged 시그널 발화.
        # 그 과정에서 lbl_value 도 갱신됨.
        self.slider.setMinimum(int(min_rad * SLIDER_SCALE))
        self.slider.setMaximum(int(max_rad * SLIDER_SCALE))
        # tick interval 도 새 범위에 맞게 — 약 10등분.
        rng = max_rad - min_rad
        self.slider.setTickInterval(max(1, int(rng / 10.0 * SLIDER_SCALE)))
        # tooltip 갱신
        self.lbl_name.setToolTip(
            f"{self.joint_name}\nrange: [{min_rad:+.3f}, {max_rad:+.3f}] rad"
        )


# ────────────────────────────────────────────────────────────────────
# 메인 윈도우
# ────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self, ros_node: RosNode):
        super().__init__()
        self.ros_node = ros_node
        self.setWindowTitle("KHUBER Pose Slider")
        self.resize(900, 700)

        # 조인트별 슬라이더 위젯 저장. OrderedDict 로 순서 보존.
        self.sliders: "OrderedDict[str, JointSliderRow]" = OrderedDict()

        self._build_ui()
        self._connect_signals()

        # 100ms 주기로 실제 위치 라벨 갱신.
        # ROS 콜백에서 직접 위젯 만지면 안 되므로 (Qt thread-safety),
        # 대신 ros_node 가 캐싱한 값을 GUI 스레드에서 읽어서 표시.
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._refresh_actual_positions)
        self._refresh_timer.start(100)

        # 시작 직후 모터 EEPROM 한계값을 자동 조회 → 슬라이더 범위 자동 설정.
        # 단, 서비스 노드가 아직 안 떠 있을 수 있으니 500ms 지연 후 시도.
        # (one-shot timer 패턴)
        QTimer.singleShot(500, self._on_reload_limits)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # ─── 상단: 다리(좌) / 팔(우) 두 그룹을 좌우로 배치 ───
        # QSplitter 로 사용자가 좌우 비율 조정 가능.
        splitter = QSplitter(Qt.Horizontal)

        # 다리 그룹
        leg_group = QGroupBox("Legs (12 joints)")
        leg_layout = QVBoxLayout(leg_group)
        for name, default in LEG_JOINTS:
            row = JointSliderRow(name, default)
            self.sliders[name] = row
            leg_layout.addWidget(row)
        leg_layout.addStretch(1)
        splitter.addWidget(leg_group)

        # 팔 그룹
        arm_group = QGroupBox("Arms (6 joints)")
        arm_layout = QVBoxLayout(arm_group)
        for name, default in ARM_JOINTS:
            row = JointSliderRow(name, default)
            self.sliders[name] = row
            arm_layout.addWidget(row)
        arm_layout.addStretch(1)
        splitter.addWidget(arm_group)

        # 다리/팔 너비 비율 (다리가 더 많으니 약간 더 넓게)
        splitter.setSizes([550, 350])
        root.addWidget(splitter, stretch=1)

        # ─── 중간: 컨트롤 버튼들 ───
        ctrl_layout = QHBoxLayout()

        self.btn_apply = QPushButton("Apply (전체 발행)")
        self.btn_apply.setStyleSheet(
            "background-color: #2d7a2d; color: white; font-weight: bold; padding: 8px;"
        )
        self.btn_apply.setToolTip(
            "모든 슬라이더 값을 /joint_goals 로 한 번에 발행.\n"
            "Profile RPM 도 함께 적용됨."
        )
        ctrl_layout.addWidget(self.btn_apply)

        # Profile RPM 입력 (Apply 시 적용될 모터 속도)
        ctrl_layout.addWidget(QLabel("RPM:"))
        self.spin_rpm = QDoubleSpinBox()
        self.spin_rpm.setRange(1.0, 200.0)
        self.spin_rpm.setValue(20.0)
        self.spin_rpm.setSingleStep(5.0)
        self.spin_rpm.setMaximumWidth(80)
        self.spin_rpm.setToolTip(
            "Apply 시 모터 profile velocity. 천천히=20, 보통=50, 빠름=100"
        )
        ctrl_layout.addWidget(self.spin_rpm)

        ctrl_layout.addSpacing(20)

        self.btn_sync_all = QPushButton("Sync All from Robot")
        self.btn_sync_all.setToolTip(
            "모든 슬라이더를 실제 모터 위치로 동기화.\n"
            "토크 OFF 후 손으로 자세 잡고 → 이 버튼 → Copy as YAML 흐름이 편함."
        )
        ctrl_layout.addWidget(self.btn_sync_all)

        self.btn_reload_limits = QPushButton("Reload Limits")
        self.btn_reload_limits.setToolTip(
            "모터 EEPROM 의 Min/Max Position Limit 를 다시 읽어 슬라이더 범위 갱신.\n"
            "Dynamixel Wizard 등으로 모터 한계를 바꾼 후 호출."
        )
        ctrl_layout.addWidget(self.btn_reload_limits)

        ctrl_layout.addSpacing(20)

        self.btn_torque_on = QPushButton("Torque ON")
        self.btn_torque_on.setStyleSheet("background-color: #d4a017; color: black;")
        ctrl_layout.addWidget(self.btn_torque_on)

        self.btn_torque_off = QPushButton("Torque OFF")
        self.btn_torque_off.setStyleSheet("background-color: #a02020; color: white;")
        self.btn_torque_off.setToolTip(
            "토크 OFF — 손으로 로봇 자세를 잡을 수 있음 (kinesthetic teaching).\n"
            "주의: 다리/팔이 무거우면 자세가 무너질 수 있으니 사전에 안전 자세 또는 받침대 필요."
        )
        ctrl_layout.addWidget(self.btn_torque_off)

        ctrl_layout.addStretch(1)
        root.addLayout(ctrl_layout)

        # ─── 하단: Phase 저장 영역 ───
        save_group = QGroupBox("Save as standup_keyframes.yaml phase")
        save_layout = QVBoxLayout(save_group)

        # phase 메타데이터 입력
        meta_layout = QHBoxLayout()
        meta_layout.addWidget(QLabel("name:"))
        self.le_phase_name = QLineEdit("new_phase")
        self.le_phase_name.setMaximumWidth(200)
        meta_layout.addWidget(self.le_phase_name)
        meta_layout.addSpacing(20)
        meta_layout.addWidget(QLabel("rpm:"))
        self.spin_phase_rpm = QDoubleSpinBox()
        self.spin_phase_rpm.setRange(1.0, 200.0)
        self.spin_phase_rpm.setValue(30.0)
        self.spin_phase_rpm.setMaximumWidth(80)
        meta_layout.addWidget(self.spin_phase_rpm)
        meta_layout.addSpacing(20)
        meta_layout.addWidget(QLabel("hold_sec:"))
        self.spin_phase_hold = QDoubleSpinBox()
        self.spin_phase_hold.setRange(0.1, 10.0)
        self.spin_phase_hold.setValue(1.0)
        self.spin_phase_hold.setSingleStep(0.1)
        self.spin_phase_hold.setMaximumWidth(80)
        meta_layout.addWidget(self.spin_phase_hold)
        meta_layout.addSpacing(20)
        self.btn_generate = QPushButton("Generate YAML")
        meta_layout.addWidget(self.btn_generate)
        self.btn_copy = QPushButton("Copy to Clipboard")
        meta_layout.addWidget(self.btn_copy)
        meta_layout.addStretch(1)
        save_layout.addLayout(meta_layout)

        # YAML 출력 영역 (편집 가능 — 사용자가 손으로 후처리도 가능)
        self.txt_yaml = QTextEdit()
        self.txt_yaml.setFontFamily("monospace")
        self.txt_yaml.setMaximumHeight(200)
        self.txt_yaml.setPlaceholderText(
            "Generate YAML 버튼을 누르면 여기에 phase 블록이 표시됩니다.\n"
            "standup_keyframes.yaml 의 forward_sequence: 아래에 그대로 붙여넣으세요."
        )
        save_layout.addWidget(self.txt_yaml)

        root.addWidget(save_group)

        # ─── 상태 바 ───
        self.statusBar().showMessage("Ready. 슬라이더 조정 후 Apply 누르세요.")

    def _connect_signals(self):
        # 메인 버튼들
        self.btn_apply.clicked.connect(self._on_apply)
        self.btn_sync_all.clicked.connect(self._on_sync_all)
        self.btn_reload_limits.clicked.connect(self._on_reload_limits)
        self.btn_torque_on.clicked.connect(lambda: self._on_torque(True))
        self.btn_torque_off.clicked.connect(lambda: self._on_torque(False))
        self.btn_generate.clicked.connect(self._on_generate_yaml)
        self.btn_copy.clicked.connect(self._on_copy_yaml)

        # 각 슬라이더의 Sync 버튼 + 토크 체크박스 시그널
        for name, row in self.sliders.items():
            # closure 캡처를 위해 default arg 트릭
            row.btn_sync.clicked.connect(lambda _=False, n=name: self._sync_one(n))
            row.torque_toggle_requested.connect(self._on_joint_torque_toggle)

    # ── 콜백들 ──────────────────────────────────────────────────────

    def _on_apply(self):
        """모든 슬라이더 값을 모아서 한 번에 발행."""
        rpm = float(self.spin_rpm.value())
        # 우리는 /joint_goals 만 발행하고, profile velocity 는 별도 토픽이 없음.
        # body_motor_controller 의 cb_set_profile_velocity 가 받는 토픽이
        # 따로 있을지 모르지만 일단 안전하게 사용자가 다른 곳에서 RPM 을
        # 미리 설정해 두는 것을 가정. (필요하면 추후 발행 추가)
        # → 우선은 RPM 입력은 단순 표시용으로 두고 향후 확장.
        # (TODO: profile velocity 토픽 발견되면 같이 발행)

        joint_dict = {name: row.get_value() for name, row in self.sliders.items()}
        try:
            self.ros_node.publish_goals(joint_dict)
            self.statusBar().showMessage(
                f"Applied {len(joint_dict)} joints (RPM={rpm:.0f} 는 별도 설정 필요)",
                3000,
            )
        except Exception as e:
            QMessageBox.warning(self, "Apply 실패", str(e))

    def _on_sync_all(self):
        """현재 실제 모터 위치로 모든 슬라이더 동기화."""
        positions = self.ros_node.get_positions()
        if not positions:
            QMessageBox.information(
                self, "Sync All",
                "/joint_states 가 아직 수신되지 않았습니다.\n"
                "body_motor_controller 가 실행 중인지 확인해 주세요."
            )
            return

        synced = 0
        missing = []
        for name, row in self.sliders.items():
            if name in positions:
                row.set_value(positions[name])
                synced += 1
            else:
                missing.append(name)

        msg = f"Synced {synced}/{len(self.sliders)} joints"
        if missing:
            msg += f" (missing: {', '.join(missing[:3])}{'...' if len(missing)>3 else ''})"
        self.statusBar().showMessage(msg, 4000)

    def _sync_one(self, joint_name: str):
        """한 조인트만 실제 값으로 동기화."""
        positions = self.ros_node.get_positions()
        if joint_name not in positions:
            self.statusBar().showMessage(
                f"{joint_name}: /joint_states 에 없음", 3000)
            return
        self.sliders[joint_name].set_value(positions[joint_name])
        self.statusBar().showMessage(
            f"{joint_name} synced to {positions[joint_name]:+.3f}", 2000)

    def _on_reload_limits(self):
        """
        모터 EEPROM 의 Position Limit 을 다시 조회해서 슬라이더 범위 갱신.

        시작 시 자동 호출 + 사용자가 버튼 누르면 수동 호출.
        서비스 호출 결과 콜백에서 각 슬라이더의 set_limits 호출.
        """
        # 호출 중임을 알림
        self.statusBar().showMessage("모터 한계값 조회 중...", 0)

        def on_done(success: bool, message: str, limits: dict):
            if not success:
                self.statusBar().showMessage(
                    f"한계값 조회 실패: {message}", 5000
                )
                return
            # 각 슬라이더에 적용
            applied = 0
            for name, row in self.sliders.items():
                if name in limits:
                    lower, upper = limits[name]
                    row.set_limits(lower, upper)
                    applied += 1
            self.statusBar().showMessage(
                f"한계값 적용: {applied}/{len(self.sliders)} joints — {message}",
                4000,
            )

        self.ros_node.call_get_joint_limits(done_cb=on_done)

    def _on_torque(self, enable: bool):
        """토크 ON/OFF 서비스 호출 (전체 본체 모터)."""
        # 응답은 비동기로 받아서 statusBar 표시.
        # 단, 콜백은 ROS 스레드에서 호출되므로 위젯 직접 만지면 안 됨.
        # → 큐에 넣거나 QMetaObject.invokeMethod 같은 게 정석이지만,
        #   여기서는 간단히 위젯 setter 가 thread-safe 한 setText 만 쓰므로
        #   직접 호출해도 Qt 가 알아서 처리해주는 경우가 많음. 안전을 위해
        #   timer 한번 거치는 식으로 갈 수도 있지만 일단 직접 호출.
        def on_done(success: bool, message: str):
            # 전체 토크 변경이 성공하면 모든 조인트 체크박스도 동기화.
            # (그렇지 않으면 GUI 상태와 실제 모터 상태가 어긋남)
            if success:
                for row in self.sliders.values():
                    row.set_torque_state(enable)
            self.statusBar().showMessage(
                f"Torque {'ON' if enable else 'OFF'} (전체): "
                f"{'OK' if success else 'FAIL'} — {message}",
                4000,
            )
        self.ros_node.call_torque(enable, done_cb=on_done)
        self.statusBar().showMessage(
            f"Torque {'ON' if enable else 'OFF'} (전체) 요청...", 2000)

    def _on_joint_torque_toggle(self, joint_name: str, want_enable: bool):
        """
        한 조인트의 체크박스가 토글됐을 때 호출. 해당 조인트만 토크 변경.

        흐름:
          1) /u2d2/torque_enable_joints 호출 (비동기)
          2) 응답 성공 → 그대로 둠 (체크박스 이미 want_enable 로 토글됨)
          3) 응답 실패 → 체크박스를 원래 상태로 원복 (set_torque_state)
        """
        def on_done(success: bool, message: str, applied: list):
            if success and joint_name in applied:
                # 정상 처리. 체크박스 상태가 곧 모터 상태와 일치.
                self.statusBar().showMessage(
                    f"{joint_name}: torque {'ON' if want_enable else 'OFF'} OK",
                    2500,
                )
            else:
                # 실패 — 체크박스를 원래 상태로 원복.
                # (현재 토글된 want_enable 의 반대값으로 되돌림)
                row = self.sliders.get(joint_name)
                if row is not None:
                    row.set_torque_state(not want_enable)
                self.statusBar().showMessage(
                    f"{joint_name}: torque 변경 실패 — {message} (체크박스 원복)",
                    5000,
                )

        self.ros_node.call_torque_joints(
            [joint_name], want_enable, done_cb=on_done,
        )
        self.statusBar().showMessage(
            f"{joint_name}: torque {'ON' if want_enable else 'OFF'} 요청...",
            1500,
        )

    def _on_generate_yaml(self):
        """현재 슬라이더 값들을 YAML phase 블록으로 출력."""
        name = self.le_phase_name.text().strip() or "unnamed"
        rpm = float(self.spin_phase_rpm.value())
        hold = float(self.spin_phase_hold.value())

        # standup_keyframes.yaml 의 phase 포맷을 그대로 따라감.
        # joints 들여쓰기는 4 spaces (yaml 내 phase 1개당 들여쓰기 일관성).
        lines = [
            f"  - name: {name}",
            f"    rpm: {rpm:.1f}",
            f"    hold_sec: {hold:.1f}",
            f"    joints:",
        ]
        # LEG_JOINTS → ARM_JOINTS 순서로 출력 (가독성)
        for joint_name, _default in ALL_JOINTS:
            val = self.sliders[joint_name].get_value()
            # rad 값을 소수점 3자리까지. 0.0 은 음수 부호 없이 깔끔하게.
            # f-string + abs() 트릭으로 -0.000 방지
            v_clean = 0.0 if abs(val) < 1e-4 else val
            lines.append(f"      {joint_name}: {v_clean:.3f}")

        yaml_text = "\n".join(lines)
        self.txt_yaml.setPlainText(yaml_text)
        self.statusBar().showMessage(
            f"YAML phase '{name}' generated. Copy 버튼으로 클립보드 복사 가능.",
            3000,
        )

    def _on_copy_yaml(self):
        """현재 YAML 텍스트 영역 내용을 클립보드에."""
        text = self.txt_yaml.toPlainText()
        if not text:
            self.statusBar().showMessage(
                "복사할 내용 없음. Generate YAML 먼저 누르세요.", 3000)
            return
        QGuiApplication.clipboard().setText(text)
        self.statusBar().showMessage("✓ Copied to clipboard.", 2000)

    def _refresh_actual_positions(self):
        """100ms 주기 — 실제 모터 위치 라벨 갱신."""
        positions = self.ros_node.get_positions()
        for name, row in self.sliders.items():
            row.update_actual(positions.get(name))

    def closeEvent(self, event):
        """창 닫을 때 깔끔하게 종료."""
        # ROS spin thread 는 daemon 이라 자동 종료되지만,
        # 명시적으로 한번 더 cleanup 시도.
        try:
            self.ros_node.destroy_node()
        except Exception:
            pass
        super().closeEvent(event)


# ────────────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────────────
def main():
    # ROS 초기화 먼저
    rclpy.init()
    ros_node = RosNode()

    # ROS spin 을 별도 스레드에서 시작
    spin_thread = RosSpinThread(ros_node)
    spin_thread.start()

    # Qt 앱
    app = QApplication(sys.argv)
    win = MainWindow(ros_node)
    win.show()

    # Qt 이벤트 루프 진입 (블로킹). 창 닫히면 빠져나옴.
    try:
        exit_code = app.exec()
    finally:
        spin_thread.stop()
        # rclpy shutdown — 이미 ok 가 False 일 수 있음
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass

    sys.exit(exit_code)


if __name__ == '__main__':
    main()