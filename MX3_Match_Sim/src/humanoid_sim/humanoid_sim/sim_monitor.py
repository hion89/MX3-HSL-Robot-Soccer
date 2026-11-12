#!/usr/bin/env python3
"""시뮬 경기 모니터 — 1 초마다 한 줄: 로봇상태 / GC 상태 / 전략 단계 / 정답 vs 측위 / 공 / 명령 / 킥.

  ROS_DOMAIN_ID=2 ros2 run humanoid_sim sim_monitor
"""
import math

import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformListener

from humanoid_behavior_interfaces.msg import GameState, StrategyState

GC_STATE = {0: "INIT", 1: "READY", 2: "SET", 3: "PLAY", 4: "FIN"}


def yaw_q(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class Monitor(Node):
    def __init__(self):
        super().__init__("sim_monitor")
        self.v = dict(robot="-", gc="-", phase="-", gt=None, ball=None, est_ball=None, cmd=(0, 0, 0),
                      kicks=0, kick_done=0)
        self.create_subscription(String, "/robot_state", lambda m: self.v.update(robot=m.data), 10)
        self.create_subscription(GameState, "/game_state", lambda m: self.v.update(gc=GC_STATE.get(m.state, m.state)), 10)
        self.create_subscription(StrategyState, "/strategy/state", self.cb_strategy, 10)
        self.create_subscription(PoseStamped, "/sim/gt/base_pose", lambda m: self.v.update(gt=m.pose), 10)
        self.create_subscription(PointStamped, "/sim/gt/ball", lambda m: self.v.update(ball=m.point), 10)
        self.create_subscription(Twist, "/cmd_vel", lambda m: self.v.update(cmd=(m.linear.x, m.linear.y, m.angular.z)), 10)
        self.create_subscription(Bool, "/kick_trigger", lambda m: self.v.update(kicks=self.v["kicks"] + int(m.data)), 10)
        self.create_subscription(String, "/sim/event", lambda m: print(f"*** {m.data}", flush=True), 10)
        self.create_subscription(Bool, "/kick_done", lambda m: self.v.update(kick_done=self.v["kick_done"] + int(m.data)), 10)
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.t0 = self.get_clock().now()
        self.create_timer(1.0, self.tick)

    def cb_strategy(self, m):
        self.v.update(phase=m.strategy_phase, est_ball=(m.ball_map_x, m.ball_map_y) if m.ball_valid else None)

    def tick(self):
        v = self.v
        t = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
        gt = "-"
        if v["gt"] is not None:
            p = v["gt"].position
            gt = f"({p.x:+.2f},{p.y:+.2f},{math.degrees(yaw_q(v['gt'].orientation)):+4.0f}°)"
        est = "-"
        try:
            tr = self.tf.lookup_transform("map", "base_link", Time())
            q = tr.transform.rotation
            est = f"({tr.transform.translation.x:+.2f},{tr.transform.translation.y:+.2f},{math.degrees(yaw_q(q)):+4.0f}°)"
        except Exception:
            pass
        ball = f"({v['ball'].x:+.2f},{v['ball'].y:+.2f})" if v["ball"] is not None else "-"
        eb = f"({v['est_ball'][0]:+.2f},{v['est_ball'][1]:+.2f})" if v["est_ball"] else "-"
        c = v["cmd"]
        print(f"{t:6.1f}s {v['robot']:>8s} GC {v['gc']:>5s} {v['phase']:>10s} | 정답 {gt} 측위 {est} | "
              f"공 {ball} 추정 {eb} | cmd ({c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f}) | 킥 {v['kicks']}/{v['kick_done']}",
              flush=True)


def main():
    rclpy.init()
    rclpy.spin(Monitor())


if __name__ == "__main__":
    main()
