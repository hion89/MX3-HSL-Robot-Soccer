#!/usr/bin/env python3
"""시뮬 경기 모니터 — 로봇마다 자기 ROS_DOMAIN_ID 에 붙어 '스택이 생각하는 것' 과 정답을 비교한다.

  match_monitor.py --scene <models/scene_4v4> [--duration 300] [--period 5] [--log out.jsonl]

로봇 이름 규약 t<팀>p<번호> (예: t35p2). 팀 좌표: 첫 팀(장면의 첫 로봇) = 월드 그대로 (+x 공격),
다른 팀 = 월드를 180° 돌린 좌표 (그 팀의 측위가 쓰는 좌표).
주기 출력: 로봇별 상태/역할/스킬, 측위 오차 (위치 m / 방향 °), 공 추정 오차, 킥 수, 넘어짐 수.
끝나면 요약 (측위 오차 평균·p90·0.5 m 초과 비율, 공 오차, 킥·넘어짐·득점) 을 출력하고 --log 에 JSON 한 줄씩.
"""
import argparse
import json
import math
import re
import statistics
import time

import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, PoseStamped
from std_msgs.msg import String

from hsl_interfaces.msg import BallEstimate, RobotPoseEstimate


def yaw_of_q(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Robot:
    def __init__(self, meta, home_team):
        self.name = meta["name"]
        m = re.match(r"t(\d+)p(\d+)", self.name)
        self.team, self.player = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        self.mirror = self.team != home_team
        self.ctx = Context()
        rclpy.init(context=self.ctx, domain_id=int(meta["domain"]))
        self.node = Node(f"match_monitor_{self.name}", context=self.ctx)
        self.exe = SingleThreadedExecutor(context=self.ctx)
        self.exe.add_node(self.node)
        self.dbg = {}
        self.pose = None
        self.ball = None
        self.gt = None
        self.gt_hist = []
        self.gt_ball = None
        self.robot_state = "NORMAL"
        self.falls = 0
        self.events = []
        n = self.node
        n.create_subscription(String, "/hsl/behavior_debug", self._dbg, 10)
        n.create_subscription(RobotPoseEstimate, "/hsl/robot_pose", lambda m: setattr(self, "pose", m), 10)
        n.create_subscription(BallEstimate, "/hsl/ball", lambda m: setattr(self, "ball", m), 10)
        n.create_subscription(PoseStamped, "/sim/gt/base_pose", self._gt, 10)
        n.create_subscription(PointStamped, "/sim/gt/ball", self._gt_ball, 10)
        n.create_subscription(String, "/robot_state", self._state, 10)
        n.create_subscription(String, "/sim/event", lambda m: self.events.append(m.data), 10)
        # 집계
        self.loc_err, self.yaw_err, self.ball_err = [], [], []
        self.skill_time = {}
        self.jumps = 0

    def _dbg(self, m):
        try:
            self.dbg = json.loads(m.data)
        except ValueError:
            pass

    def _to_team(self, x, y, th=0.0):
        return (-x, -y, wrap(th + math.pi)) if self.mirror else (x, y, th)

    def _gt(self, m):
        self.gt = self._to_team(m.pose.position.x, m.pose.position.y, yaw_of_q(m.pose.orientation))
        self.gt_hist.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, *self.gt))
        if len(self.gt_hist) > 50:
            self.gt_hist.pop(0)

    def gt_at(self, t):
        """정답을 측위 출력 시각으로 보간 (정답은 10 Hz 라 최신값과 비교하면 회전 중 수~십수 ° 착시)."""
        h = self.gt_hist
        for i in range(len(h) - 1, 0, -1):
            if h[i - 1][0] <= t <= h[i][0] and h[i][0] - h[i - 1][0] < 0.3:
                a = (t - h[i - 1][0]) / max(h[i][0] - h[i - 1][0], 1e-6)
                dth = wrap(h[i][3] - h[i - 1][3])
                return (h[i - 1][1] + a * (h[i][1] - h[i - 1][1]), h[i - 1][2] + a * (h[i][2] - h[i - 1][2]),
                        wrap(h[i - 1][3] + a * dth))
        return self.gt

    def pose_err(self):
        if self.pose is None or self.gt is None:
            return None, None
        g = self.gt_at(self.pose.header.stamp.sec + self.pose.header.stamp.nanosec * 1e-9)
        return math.hypot(self.pose.x - g[0], self.pose.y - g[1]), math.degrees(abs(wrap(self.pose.theta - g[2])))

    def _gt_ball(self, m):
        self.gt_ball = self._to_team(m.point.x, m.point.y)[:2]

    def _state(self, m):
        if self.robot_state == "NORMAL" and m.data != "NORMAL":
            self.falls += 1
        self.robot_state = m.data

    def sample(self, dt):
        e, ey = self.pose_err()
        if e is not None and self.dbg.get("state") in ("READY", "SET", "PLAYING"):
            self.loc_err.append(e)
            self.yaw_err.append(ey)
        be = None
        if self.ball is not None and self.ball.valid and self.ball.time_since_seen < 1.0 and self.gt_ball:
            be = math.hypot(self.ball.field_x - self.gt_ball[0], self.ball.field_y - self.gt_ball[1])
            self.ball_err.append(be)
        sk = re.sub(r"\(.*\)", "", str(self.dbg.get("skill", "?")))
        self.skill_time[sk] = self.skill_time.get(sk, 0.0) + dt
        return e, be

    def line(self):
        d = self.dbg
        be = None
        e, yaw = self.pose_err()
        if self.ball is not None and self.ball.valid and self.gt_ball and self.ball.time_since_seen < 1.0:
            be = math.hypot(self.ball.field_x - self.gt_ball[0], self.ball.field_y - self.gt_ball[1])
        gt = "      ?      " if self.gt is None else f"({self.gt[0]:+.1f},{self.gt[1]:+.1f})"
        fmt = lambda v, f: "  -  " if v is None else format(v, f)  # noqa: E731
        q = "" if self.pose is None else "PRS"[min(self.pose.quality, 2)]
        return (f"{self.name:6s} {str(d.get('state', '-'))[:4]:4s} {str(d.get('role', '-'))[:6]:6s} "
                f"{'A' if d.get('active') == self.player else ' '} {str(d.get('skill', '-'))[:24]:24s} "
                f"gt{gt:13s} loc {fmt(e, '5.2f')}m {fmt(yaw, '4.0f')}° {q} ball {fmt(be, '4.2f')} "
                f"kick {d.get('kicks', 0)} fall {self.falls} {self.robot_state if self.robot_state != 'NORMAL' else ''}")

    def summary(self):
        def stats(v):
            if not v:
                return None
            s = sorted(v)
            return dict(mean=round(statistics.fmean(s), 3), p50=round(s[len(s) // 2], 3),
                        p90=round(s[int(len(s) * 0.9)], 3), n=len(s))
        return dict(robot=self.name, team=self.team, player=self.player, loc=stats(self.loc_err),
                    loc_over_0_5=round(sum(e > 0.5 for e in self.loc_err) / max(len(self.loc_err), 1), 3),
                    yaw=stats(self.yaw_err), ball=stats(self.ball_err), kicks=self.dbg.get("kicks", 0),
                    falls=self.falls, skills={k: round(v, 1) for k, v in sorted(self.skill_time.items())})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--duration", type=float, default=0.0, help="0 = Ctrl-C 까지")
    ap.add_argument("--period", type=float, default=5.0)
    ap.add_argument("--log", default="")
    args = ap.parse_args()
    meta = json.load(open(args.scene + ".json"))
    home = int(re.match(r"t(\d+)", meta["robots"][0]["name"]).group(1))
    robots = [Robot(r, home) for r in meta["robots"]]
    logf = open(args.log, "a") if args.log else None
    t0 = time.monotonic()
    next_print = next_sample = 0.0
    events_seen = 0
    try:
        while args.duration <= 0 or time.monotonic() - t0 < args.duration:
            for r in robots:
                r.exe.spin_once(timeout_sec=0.0)
            now = time.monotonic() - t0
            if now >= next_sample:
                next_sample += 0.5
                for r in robots:
                    e, be = r.sample(0.5)
                    if logf and int(now * 2) % 2 == 0:   # 1 s 마다 시계열
                        p, g = r.pose, r.gt
                        logf.write(json.dumps(dict(
                            t=round(now, 1), robot=r.name, state=r.dbg.get("state"), skill=r.dbg.get("skill"),
                            gt=None if g is None else [round(g[0], 2), round(g[1], 2), round(math.degrees(g[2]))],
                            est=None if p is None else [round(p.x, 2), round(p.y, 2), round(math.degrees(p.theta))],
                            q=None if p is None else p.quality, loc=None if e is None else round(e, 3),
                            yaw=None if r.pose_err()[1] is None else round(r.pose_err()[1], 1),
                            ball=None if be is None else round(be, 3), rs=r.robot_state), ensure_ascii=False) + "\n")
            evs = robots[0].events
            while events_seen < len(evs):
                print(f"[{now:6.1f}s] ★ {evs[events_seen]}", flush=True)
                if logf:
                    logf.write(json.dumps(dict(t=round(now, 1), event=evs[events_seen])) + "\n")
                events_seen += 1
            if now >= next_print:
                next_print += args.period
                print(f"── t={now:6.1f}s " + "─" * 100, flush=True)
                for r in robots:
                    print("  " + r.line(), flush=True)
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    print("═" * 40 + " 요약 " + "═" * 40, flush=True)
    for r in robots:
        s = r.summary()
        print(json.dumps(s, ensure_ascii=False), flush=True)
        if logf:
            logf.write(json.dumps(dict(summary=s), ensure_ascii=False) + "\n")
    goals = [e for e in robots[0].events if e.startswith("GOAL")]
    print(f"득점 이벤트 {len(goals)}: " + " | ".join(goals), flush=True)
    if logf:
        logf.close()


if __name__ == "__main__":
    main()
