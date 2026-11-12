#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from humanoid_behavior_interfaces.srv import RobotInit

class RobotInitClient(Node):
    def __init__(self):
        super().__init__("robot_init_client")

        self.declare_parameter("service_name", "/u2d2/robot_init")
        self.declare_parameter("init_rpm", 100.0)
        self.declare_parameter("final_rpm", 200.0)
        self.declare_parameter("pos_tolerance_rad", 0.03)
        self.declare_parameter("settle_count", 10)
        self.declare_parameter("timeout_sec", 20.0)

        srv_name = self.get_parameter("service_name").value
        self.cli = self.create_client(RobotInit, srv_name)
        if not self.cli.wait_for_service(timeout_sec=2.0):
            raise RuntimeError(f"service not available: {srv_name}")

        req = RobotInit.Request()
        req.init_rpm = float(self.get_parameter("init_rpm").value)
        req.final_rpm = float(self.get_parameter("final_rpm").value)
        req.pos_tolerance_rad = float(self.get_parameter("pos_tolerance_rad").value)
        req.settle_count = int(self.get_parameter("settle_count").value)
        req.timeout_sec = float(self.get_parameter("timeout_sec").value)

        self.get_logger().info("Calling robot_init service...")
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=req.timeout_sec + 5.0)

        resp = future.result()
        if resp is None:
            self.get_logger().error("robot_init service call failed")
        else:
            self.get_logger().info(f"robot_init: success={resp.success} msg='{resp.message}'")

        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    RobotInitClient()
    rclpy.spin(rclpy.get_default_context())  # shutdown already called in node
