#!/usr/bin/env python3

import csv
import math
import os
import time
from datetime import datetime

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Vector3Stamped


class PolicyLogger(Node):

    def __init__(self):
        super().__init__("policy_logger")

        # -------- parameters --------
        self.declare_parameter("rate_hz", 25.0)
        self.declare_parameter("output_dir", "~/policy_logs")
        self.declare_parameter("flush_every_n_rows", 25)

        rate = float(self.get_parameter("rate_hz").value)
        self.period = 1.0 / rate
        self.flush_every_n_rows = int(self.get_parameter("flush_every_n_rows").value)

        out_dir = os.path.expanduser(self.get_parameter("output_dir").value)
        os.makedirs(out_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(out_dir, f"policy_log_{timestamp}.csv")

        self.get_logger().info(f"CSV logging to: {self.csv_path}")

        # -------- latest data --------
        self.joint_state = None
        self.imu_msg = None
        self.projected_gravity_msg = None

        # -------- save state --------
        self.header_written = False
        self.row_count = 0
        self._closed = False
        self._joint_name_order = None

        # 벽시계 시작시간
        self.start_time = None

        # -------- subscribers --------
        self.create_subscription(
            JointState,
            "/joint_states",
            self.cb_joint,
            10,
        )

        self.create_subscription(
            Imu,
            "/imu/data",
            self.cb_imu,
            10,
        )

        self.create_subscription(
            Vector3Stamped,
            "/imu/projected_gravity",
            self.cb_gravity,
            10,
        )

        # -------- csv --------
        self.file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)

        # -------- timer --------
        self.create_timer(self.period, self.timer_cb)

    # ------------------------------

    def cb_joint(self, msg):
        self.joint_state = msg
        if self._joint_name_order is None:
            self._joint_name_order = list(msg.name)

    def cb_imu(self, msg):
        self.imu_msg = msg

    def cb_gravity(self, msg):
        self.projected_gravity_msg = msg

    # ------------------------------

    def get_ang_vel(self):
        if self.imu_msg is None:
            return [math.nan, math.nan, math.nan]

        return [
            self.imu_msg.angular_velocity.x,
            self.imu_msg.angular_velocity.y,
            self.imu_msg.angular_velocity.z,
        ]

    def get_gravity(self):
        if self.projected_gravity_msg is None:
            return [math.nan, math.nan, math.nan]

        return [
            self.projected_gravity_msg.vector.x,
            self.projected_gravity_msg.vector.y,
            self.projected_gravity_msg.vector.z,
        ]

    # ------------------------------

    def write_header(self):

        if self.header_written:
            return

        if self._joint_name_order is None:
            return

        header = ["time_ms"]

        for name in self._joint_name_order:
            header.append(f"{name}_pos")

        for name in self._joint_name_order:
            header.append(f"{name}_vel")

        header += [
            "imu_wx",
            "imu_wy",
            "imu_wz",
            "g_x",
            "g_y",
            "g_z",
        ]

        self.writer.writerow(header)
        self.file.flush()
        self.header_written = True

    # ------------------------------

    def timer_cb(self):

        if self.joint_state is None:
            return

        self.write_header()

        js = self.joint_state
        n = len(js.name)

        if self.start_time is None:
            self.start_time = time.time()

        now = time.time()
        elapsed_ms = (now - self.start_time) * 1000.0

        row = [elapsed_ms]

        row.extend(js.position[:n])
        row.extend(js.velocity[:n])

        row.extend(self.get_ang_vel())
        row.extend(self.get_gravity())

        self.writer.writerow(row)

        self.row_count += 1

        if self.row_count % self.flush_every_n_rows == 0:
            self.file.flush()

    # ------------------------------

    def destroy_node(self):

        if not self._closed:
            try:
                self.file.flush()
                os.fsync(self.file.fileno())
                self.file.close()
            except:
                pass

        self._closed = True

        super().destroy_node()


# ------------------------------

def main(args=None):

    rclpy.init(args=args)

    node = PolicyLogger()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        try:
            node.destroy_node()
        except:
            pass

        try:
            if rclpy.ok():
                rclpy.shutdown()
        except:
            pass


if __name__ == "__main__":
    main()