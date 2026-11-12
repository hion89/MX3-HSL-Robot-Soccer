#!/usr/bin/env python3
# standup_node.py
#
# standup_keyframes.yaml의 키프레임 시퀀스를 재생하는 독립 ROS 노드
# actor_onnx_real_runner.py와 같은 프로젝트에 넣어서 사용
#
# 기능:
#   - /joint_states 구독 (현재 관절 위치/속도)
#   - /imu/projected_gravity 구독 (중력 벡터)
#   - /imu/data 구독 (IMU raw)
#   - /joint_goals 발행 (목표 관절 위치)
#   - 매 tick마다 CSV 로깅 (sim-vs-real 비교용)
#
# 사용법:
#   ros2 run humanoid_policy_runner standup_node \
#     --ros-args \
#     -p keyframe_yaml:=/path/to/standup_keyframes.yaml \
#     -p direction:=forward \
#     -p auto_start:=true
#
# CSV 출력: ~/policy_logs/standup_real_YYYYMMDD_HHMMSS.csv

import os
import csv
import time
import math
from datetime import datetime
from typing import Optional

import yaml
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Vector3Stamped

try:
    from ament_index_python.packages import get_package_share_directory
except Exception:
    get_package_share_directory = None


JOINT_ORDER = [
    "r_pelvis_pitch_joint", "r_pelvis_roll_joint", "r_pelvis_yaw_joint",
    "r_knee_pitch_joint", "r_ankle_pitch_joint", "r_ankle_roll_joint",
    "l_pelvis_pitch_joint", "l_pelvis_roll_joint", "l_pelvis_yaw_joint",
    "l_knee_pitch_joint", "l_ankle_pitch_joint", "l_ankle_roll_joint",
    "r_shoulder_pitch_joint", "r_shoulder_roll_joint", "r_elbow_pitch_joint",
    "l_shoulder_pitch_joint", "l_shoulder_roll_joint", "l_elbow_pitch_joint",
]


class StandUpNode(Node):
    def __init__(self):
        super().__init__("standup_node")

        # ── 파라미터 ──
        # 기본 경로: get_package_share_directory (launch 파일과 동일)
        default_yaml = ""
        if get_package_share_directory is not None:
            try:
                default_yaml = os.path.join(
                    get_package_share_directory("humanoid_serial_bridge"),
                    "config", "standup_keyframes.yaml",
                )
                self.get_logger().info(f"keyframe_yaml resolved: {default_yaml}")
                if not os.path.isfile(default_yaml):
                    self.get_logger().warn(f"File not found at share path: {default_yaml}")
                    default_yaml = ""
            except Exception as e:
                self.get_logger().warn(f"get_package_share_directory failed: {e}")

        # fallback: 소스 경로
        if not default_yaml:
            fallback = os.path.expanduser(
                "~/ros2_ws/src/humanoid_serial_bridge/config/standup_keyframes.yaml"
            )
            if os.path.isfile(fallback):
                default_yaml = fallback
                self.get_logger().warn(f"Using fallback source path: {fallback}")

        self.declare_parameter("keyframe_yaml", default_yaml)
        self.declare_parameter("direction", "forward")       # forward / backward
        self.declare_parameter("auto_start", True)           # True면 시작 즉시 실행
        self.declare_parameter("start_delay_sec", 2.0)       # 시작 전 대기 (센서 안정화)
        self.declare_parameter("control_rate_hz", 50.0)      # 제어 주기 (bridge와 동일)
        self.declare_parameter("enable_csv_log", True)
        self.declare_parameter("csv_output_dir", "~/policy_logs")

        yaml_path = os.path.expanduser(
            str(self.get_parameter("keyframe_yaml").value)
        )
        direction = str(self.get_parameter("direction").value)
        self.auto_start = bool(self.get_parameter("auto_start").value)
        self.start_delay = float(self.get_parameter("start_delay_sec").value)
        control_hz = float(self.get_parameter("control_rate_hz").value)
        self.enable_csv = bool(self.get_parameter("enable_csv_log").value)
        self.csv_dir = os.path.expanduser(
            str(self.get_parameter("csv_output_dir").value)
        )

        # ── 키프레임 로드 ──
        if not yaml_path or not os.path.isfile(yaml_path):
            raise RuntimeError(f"keyframe_yaml not found: {yaml_path}")

        self.sequence = self._load_yaml(yaml_path, direction)
        self.get_logger().info(
            f"Loaded {len(self.sequence)} phases from {yaml_path} ({direction})"
        )
        for i, (name, rpm, hold, joints) in enumerate(self.sequence):
            self.get_logger().info(
                f"  [{i}] {name}  rpm={rpm:.0f}  hold={hold:.1f}s  "
                f"joints={len(joints)}"
            )

        # ── 상태 ──
        self.last_js: Optional[JointState] = None
        self.last_imu: Optional[Imu] = None
        self.last_gravity: Optional[Vector3Stamped] = None

        self.phase_idx = -1
        self.phase_name = "idle"
        self.phase_start_time = 0.0
        self.current_hold_sec = 0.0
        self.current_targets = {}   # {joint_name: rad}

        self.running = False
        self.done = False
        self.node_start_time = time.time()
        self.tick_count = 0

        # ── CSV ──
        self._csv_file = None
        self._csv_writer = None
        self._csv_tick = 0

        # ── ROS 구독/발행 ──
        self.create_subscription(JointState, "/joint_states", self.cb_js, 20)
        self.create_subscription(Imu, "/imu/data", self.cb_imu, 20)
        self.create_subscription(
            Vector3Stamped, "/imu/projected_gravity", self.cb_grav, 20
        )
        self.goal_pub = self.create_publisher(JointState, "/joint_goals", 10)

        # ── 제어 타이머 ──
        period = 1.0 / max(1.0, control_hz)
        self.create_timer(period, self.control_tick)

        self.get_logger().info(
            f"StandUpNode ready: rate={control_hz:.0f}Hz, "
            f"auto_start={self.auto_start}, delay={self.start_delay:.1f}s"
        )

    # ──────────────────────────────────────
    # YAML 로드
    # ──────────────────────────────────────
    @staticmethod
    def _load_yaml(path: str, direction: str) -> list:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        key = f"{direction}_sequence"
        seq_data = data.get(key, [])
        if not seq_data:
            raise RuntimeError(f"'{key}' not found in {path}")

        sequence = []
        for entry in seq_data:
            name = str(entry.get("name", f"phase_{len(sequence)}"))
            rpm = float(entry.get("rpm", 60.0))
            hold = float(entry.get("hold_sec", 1.0))
            joints = {str(k): float(v) for k, v in entry.get("joints", {}).items()}
            sequence.append((name, rpm, hold, joints))

        return sequence

    # ──────────────────────────────────────
    # 콜백
    # ──────────────────────────────────────
    def cb_js(self, msg: JointState):
        self.last_js = msg

    def cb_imu(self, msg: Imu):
        self.last_imu = msg

    def cb_grav(self, msg: Vector3Stamped):
        self.last_gravity = msg

    # ──────────────────────────────────────
    # 제어 루프
    # ──────────────────────────────────────
    def control_tick(self):
        if self.done:
            return

        # 센서 대기
        if self.last_js is None:
            return

        elapsed = time.time() - self.node_start_time

        # 시작 전 대기
        if self.auto_start and not self.running and elapsed >= self.start_delay:
            self._start_sequence()

        if not self.running:
            return

        now = time.time()

        # 시간 기반 phase 전환
        phase_elapsed = now - self.phase_start_time
        if phase_elapsed >= self.current_hold_sec:
            next_idx = self.phase_idx + 1
            if next_idx < len(self.sequence):
                self._go_phase(next_idx)
            else:
                self.get_logger().info("Standup sequence complete!")
                self.running = False
                self.done = True
                self._close_csv()
                return

        # 목표 발행
        self._publish_goals()

        # CSV 로깅
        self._log_csv()

        self.tick_count += 1

    # ──────────────────────────────────────
    # 시퀀스 제어
    # ──────────────────────────────────────
    def _start_sequence(self):
        self.get_logger().info("Starting standup sequence...")
        self.running = True
        self.done = False

        if self.enable_csv:
            self._open_csv()

        self._go_phase(0)

    def _go_phase(self, idx: int):
        name, rpm, hold_sec, joints = self.sequence[idx]

        self.phase_idx = idx
        self.phase_name = name
        self.current_hold_sec = hold_sec
        self.current_targets = joints
        self.phase_start_time = time.time()

        self.get_logger().info(
            f"Phase [{idx}/{len(self.sequence)-1}] {name}  "
            f"rpm={rpm:.0f}  hold={hold_sec:.1f}s  "
            f"joints={len(joints)}"
        )

    def _publish_goals(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()

        for jn in JOINT_ORDER:
            if jn in self.current_targets:
                msg.name.append(jn)
                msg.position.append(float(self.current_targets[jn]))

        self.goal_pub.publish(msg)

    # ──────────────────────────────────────
    # CSV 로깅
    # ──────────────────────────────────────
    def _open_csv(self):
        os.makedirs(self.csv_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.csv_dir, f"standup_real_{ts}.csv")

        self._csv_file = open(path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_tick = 0

        header = ["tick", "time_sec", "phase_name", "phase_index"]

        # IMU
        header += ["imu_wx", "imu_wy", "imu_wz"]
        header += ["imu_ax", "imu_ay", "imu_az"]

        # gravity
        header += ["grav_x", "grav_y", "grav_z"]

        # pitch/roll from gravity
        header += ["pitch_deg", "roll_deg"]

        # q_target, q_actual, q_error, qd_actual
        for jn in JOINT_ORDER:
            header.append(f"q_target_{jn}")
        for jn in JOINT_ORDER:
            header.append(f"q_actual_{jn}")
        for jn in JOINT_ORDER:
            header.append(f"q_error_{jn}")
        for jn in JOINT_ORDER:
            header.append(f"qd_actual_{jn}")

        self._csv_writer.writerow(header)
        self.get_logger().info(f"CSV logging: {path} ({len(header)} columns)")

    def _log_csv(self):
        if self._csv_writer is None or self.last_js is None:
            return

        elapsed = time.time() - self.node_start_time

        # 현재 관절 상태
        cur_pos = {n: p for n, p in zip(self.last_js.name, self.last_js.position)}
        cur_vel = {}
        if self.last_js.velocity:
            cur_vel = {n: v for n, v in zip(self.last_js.name, self.last_js.velocity)}

        row = [self._csv_tick, f"{elapsed:.4f}", self.phase_name, self.phase_idx]

        # IMU
        if self.last_imu is not None:
            row += [
                f"{self.last_imu.angular_velocity.x:.6f}",
                f"{self.last_imu.angular_velocity.y:.6f}",
                f"{self.last_imu.angular_velocity.z:.6f}",
                f"{self.last_imu.linear_acceleration.x:.6f}",
                f"{self.last_imu.linear_acceleration.y:.6f}",
                f"{self.last_imu.linear_acceleration.z:.6f}",
            ]
        else:
            row += [""] * 6

        # gravity + pitch/roll
        if self.last_gravity is not None:
            gx = self.last_gravity.vector.x
            gy = self.last_gravity.vector.y
            gz = self.last_gravity.vector.z
            row += [f"{gx:.6f}", f"{gy:.6f}", f"{gz:.6f}"]
            pitch = math.degrees(math.atan2(gx, -gz))
            roll = math.degrees(math.atan2(gy, -gz))
            row += [f"{pitch:.3f}", f"{roll:.3f}"]
        else:
            row += [""] * 5

        # q_target
        for jn in JOINT_ORDER:
            row.append(f"{self.current_targets.get(jn, 0.0):.6f}")

        # q_actual
        for jn in JOINT_ORDER:
            row.append(f"{cur_pos.get(jn, 0.0):.6f}")

        # q_error (actual - target)
        for jn in JOINT_ORDER:
            actual = cur_pos.get(jn, 0.0)
            target = self.current_targets.get(jn, 0.0)
            row.append(f"{actual - target:.6f}")

        # qd_actual
        for jn in JOINT_ORDER:
            row.append(f"{cur_vel.get(jn, 0.0):.6f}")

        self._csv_writer.writerow(row)
        self._csv_tick += 1

        if self._csv_tick % 25 == 0:
            self._csv_file.flush()

    def _close_csv(self):
        if self._csv_file is None:
            return
        try:
            self._csv_file.flush()
            os.fsync(self._csv_file.fileno())
            self._csv_file.close()
            self.get_logger().info(f"CSV closed: {self._csv_tick} rows")
        except Exception:
            pass
        self._csv_file = None
        self._csv_writer = None

    # ──────────────────────────────────────
    # 종료
    # ──────────────────────────────────────
    def destroy_node(self):
        self._close_csv()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = StandUpNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()