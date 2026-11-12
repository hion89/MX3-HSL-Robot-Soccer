#!/usr/bin/env python3
"""팀 통신 — B-Human TeamMessageHandler / TeamData 대응 (단순화).

보내기 (send_rate_hz, 기본 2 Hz): 내 pose·validity, 공(필드 좌표·속도·나이), 공 도달 예상 시간,
  위치 역할, 공을 다루는 중인지, 넘어짐·페널티.  UDP 브로드캐스트, 포트 = 10000 + 팀 번호 (RoboCup 관례).
받기: 같은 팀·다른 선수 번호만. timeout 안에 받은 팀원을 /hsl/team_state 로 10 Hz 발행.

패킷 (리틀 엔디언, 51 B):  'HSLT' ver team player flags role | x y θ validity | bx by bvx bvy bage | ttb | seq
  flags: bit0 penalized, bit1 fallen, bit2 ball_valid, bit3 playing_ball
메시지 예산: HSL 2026 은 팀당 게임당 12,000 개 (GC 패킷 v20 messageBudget). 4 대 × 2 Hz × 약 20 분 ≈ 9,600 개.
4 Hz 면 19,200 개로 경기 중반에 초과.
즉시 송신: 공 담당·넘어짐·페널티 플래그가 바뀌거나 공 도달 예상 시간이 event_ttb_delta 넘게 바뀌면 (공이 차여 굴러감)
  다음 주기를 기다리지 않고 바로 보내고, 주기 시계를 거기서 다시 센다. 토큰 버킷(event_rate_hz, event_burst)으로 제한해
  평균 ≤ send_rate_hz + event_rate_hz (2 + 0.25 Hz → 4 대 20 분 10,800 개).
  근거 (2026-10-31 시뮬 4v4 A/B, 6 판): 고정 2 Hz 는 4 Hz 보다 넘어짐 2.8 배 (킥 중 넘어짐 13.7 % vs 5.6 %).
  팀원 충돌·위치 오차·담당 공백은 차이가 없었고, 상대와 공에 동시에 도착하는 킥이 32 % vs 22 % 로 늘었다
  → 공이 굴러간 뒤 공 담당 재선정이 늦는 탓으로 보고 바뀌는 순간만 즉시 보낸다.
시뮬에서는 8 대가 한 PC 라서 SO_REUSEADDR + 브로드캐스트로 같은 포트를 같이 받는다 (GC 수신과 같은 방식).
"""
import math
import socket
import struct
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from hsl_interfaces.msg import BallEstimate, RobotPoseEstimate, StrategyState, Teammate, TeamState
from humanoid_behavior_interfaces.msg import GameState

FMT = "<4sBBBBB4f5ffH"
SIZE = struct.calcsize(FMT)
MAGIC = b"HSLT"
VERSION = 1


class TeamComm(Node):
    def __init__(self):
        super().__init__("hsl_team_comm")
        self.team = int(self.declare_parameter("team_number", 35).value)
        self.player = int(self.declare_parameter("player_number", 1).value)
        port = int(self.declare_parameter("port", 0).value) or 10000 + self.team
        self.addr = (str(self.declare_parameter("broadcast_address", "255.255.255.255").value), port)
        rate = float(self.declare_parameter("send_rate_hz", 2.0).value)
        self.timeout = float(self.declare_parameter("teammate_timeout", 2.0).value)
        self.period = 1.0 / rate
        self.event_send = bool(self.declare_parameter("event_send", True).value)
        self.event_ttb_delta = float(self.declare_parameter("event_ttb_delta", 1.0).value)
        self.event_rate = float(self.declare_parameter("event_rate_hz", 0.25).value)
        self.event_burst = float(self.declare_parameter("event_burst", 3.0).value)
        self.event_min_gap = float(self.declare_parameter("event_min_gap", 0.1).value)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock.bind(("0.0.0.0", port))
        self.sock.setblocking(False)

        self.pose = None
        self.ball = None
        self.strategy = None
        self.penalized = False
        self.fallen = False
        self.seq = 0
        self.mates = {}   # player → (Teammate, 받은 시각 s)
        self.last_send = -1e9          # 단조 시계 [s]
        self.last_sent = None          # (flags 중 penalized·fallen·playing_ball, ttb) — 즉시 송신 판단용
        self.tokens = self.event_burst
        self.last_tick = time.monotonic()
        self.n_periodic = self.n_event = 0
        self.count_t0 = time.monotonic()

        self.create_subscription(RobotPoseEstimate, "/hsl/robot_pose", lambda m: setattr(self, "pose", m), 10)
        self.create_subscription(BallEstimate, "/hsl/ball", lambda m: setattr(self, "ball", m), 10)
        self.create_subscription(StrategyState, "/hsl/strategy_state", lambda m: setattr(self, "strategy", m), 10)
        self.create_subscription(GameState, "/game_state", self._on_gc, 10)
        self.create_subscription(String, "/robot_state", lambda m: setattr(self, "fallen", m.data != "NORMAL"), 10)
        self.pub = self.create_publisher(TeamState, "/hsl/team_state", 10)
        self.create_timer(0.05, self._tick)
        self.create_timer(60.0, self._report)
        self.create_timer(0.02, self._receive)
        self.create_timer(0.1, self._publish)
        self.get_logger().info(f"team comm: team {self.team} player {self.player} → {self.addr[0]}:{port}")

    def _on_gc(self, m):
        i = self.player - 1
        self.penalized = bool(m.team_found and 0 <= i < len(m.penalties) and m.penalties[i] != 0)

    def _state(self):
        """즉시 송신 판단에 쓰는 내 상태: (penalized, fallen, playing_ball), 공 도달 예상 시간."""
        s = self.strategy
        return (self.penalized, self.fallen, bool(s is not None and s.active)), \
            (min(s.my_time_to_ball, 1e3) if s is not None else 1e3)

    def _tick(self):
        now = time.monotonic()
        self.tokens = min(self.event_burst, self.tokens + (now - self.last_tick) * self.event_rate)
        self.last_tick = now
        if now - self.last_send >= self.period:
            self.n_periodic += 1
            self._send(now)
            return
        if not self.event_send or self.last_sent is None or now - self.last_send < self.event_min_gap or self.tokens < 1.0:
            return
        flags, ttb = self._state()
        sent_flags, sent_ttb = self.last_sent
        if flags != sent_flags or abs(ttb - sent_ttb) > self.event_ttb_delta:
            self.tokens -= 1.0
            self.n_event += 1
            self._send(now)

    def _report(self):
        el = time.monotonic() - self.count_t0
        n = self.n_periodic + self.n_event
        self.get_logger().info(f"송신 {n} 개 / {el:.0f} s (주기 {self.n_periodic}, 즉시 {self.n_event}) = {n / el:.2f} Hz, "
                               f"4 대 20 분 환산 {n / el * 4 * 1200:.0f} 개 (예산 12,000)")

    def _send(self, now):
        self.last_send = now
        self.last_sent = self._state()
        p, b, s = self.pose, self.ball, self.strategy
        flags = (1 if self.penalized else 0) | (2 if self.fallen else 0)
        x = y = th = val = 0.0
        if p is not None:
            x, y, th, val = p.x, p.y, p.theta, p.validity
        bx = by = bvx = bvy = 0.0
        bage = 1e3
        if b is not None and b.valid:
            flags |= 4
            bx, by, bvx, bvy, bage = b.field_x, b.field_y, b.vel_x, b.vel_y, b.time_since_seen
        ttb, role = 1e3, 0
        if s is not None:
            ttb, role = s.my_time_to_ball, s.position_role
            if s.active:
                flags |= 8
        self.seq = (self.seq + 1) & 0xFFFF
        data = struct.pack(FMT, MAGIC, VERSION, self.team & 0xFF, self.player & 0xFF, flags, role,
                           x, y, th, val, bx, by, bvx, bvy, min(bage, 1e3), min(ttb, 1e3), self.seq)
        try:
            self.sock.sendto(data, self.addr)
        except OSError as e:
            self.get_logger().warn(f"송신 실패: {e}", throttle_duration_sec=5.0)

    def _receive(self):
        now = self.get_clock().now()
        while True:
            try:
                data, _ = self.sock.recvfrom(256)
            except (BlockingIOError, InterruptedError):
                break
            except OSError:
                break
            if len(data) != SIZE:
                continue
            (magic, ver, team, player, flags, role, x, y, th, val,
             bx, by, bvx, bvy, bage, ttb, _seq) = struct.unpack(FMT, data)
            if magic != MAGIC or ver != VERSION or team != self.team or player == self.player:
                continue
            if not all(math.isfinite(v) for v in (x, y, th, val, bx, by, bvx, bvy, bage, ttb)):
                continue
            t = Teammate()
            t.received = now.to_msg()
            t.player_number = player
            t.penalized, t.fallen = bool(flags & 1), bool(flags & 2)
            t.ball_valid, t.playing_ball = bool(flags & 4), bool(flags & 8)
            t.x, t.y, t.theta, t.pose_validity = x, y, th, val
            t.ball_x, t.ball_y, t.ball_vel_x, t.ball_vel_y, t.ball_age = bx, by, bvx, bvy, bage
            t.time_to_ball, t.position_role = ttb, role
            self.mates[player] = (t, now.nanoseconds * 1e-9)

    def _publish(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        msg = TeamState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.my_player_number = self.player
        for player in sorted(self.mates):
            t, at = self.mates[player]
            if now - at <= self.timeout:
                msg.teammates.append(t)
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = TeamComm()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
