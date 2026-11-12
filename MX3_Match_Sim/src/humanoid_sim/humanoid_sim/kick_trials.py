#!/usr/bin/env python3
"""킥오프 정렬 튜닝 비교 — 판마다 시뮬+스택을 새로 띄우고 킥오프 1 회를 잰다.

  ~/sim_ws/src/humanoid_sim/scripts/run_trials.sh --configs base,lowmin,lowmin_tol --trials 8

한 판:
  1) 로봇을 센터 공 뒤쪽(우리 진영) 무작위 위치·방향에 세운다 (설정끼리 같은 배치 사용)
  2) 스택 기동 12 s → START → /initialpose = 정답 (재진입 시드 대신, 정렬만 보려고)
  3) visual servo 파라미터 설정 → GC SET 1 s → PLAYING
  4) 첫 /kick_trigger 까지 시간, 그 순간 측위 오차, 3.5 s 뒤 공 방향(골 중심 대비)을 기록. 45 s 안에 못 차면 실패.
"""
import argparse
import csv
import json
import math
import os
import signal
import subprocess
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from rcl_interfaces.msg import Parameter as ParamMsg, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

# servo: /visual_servo_kick_node 실행 중 파라미터, launch: sim_robot.launch.py 인자
_LOWMIN = {"vx_min": 0.08, "vx_back_min": 0.08, "vy_min": 0.08, "wz_min": 0.15}
CONFIGS = {
    "base": dict(servo={}, launch=[]),
    "lowmin": dict(servo=_LOWMIN, launch=[]),
    "lowmin_tol": dict(servo=dict(_LOWMIN, kick_pos_tol_x=0.10), launch=[]),
    "lowmin_nolock": dict(servo=_LOWMIN, launch=["neck_yaw_lock_deg:=-71"]),
    "lowmin_nolock_fix": dict(servo=_LOWMIN, launch=["neck_yaw_lock_deg:=-71"], env={"SIM_SERVO_FIX": "1"}),
}
GOAL = np.array([4.5, 0.0])
RECORD = False
WS = os.environ.get("SIM_WS") or os.path.expanduser("~/sim_ws")   # run_trials.sh (sim_env.sh) 가 넣어 준다
SCRIPTS = os.path.join(WS, "src/humanoid_sim/scripts")
SCENE = os.path.join(WS, "install/humanoid_sim/share/humanoid_sim/models/scene_1v0")


def yaw_q(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class Watcher(Node):
    def __init__(self):
        super().__init__("kick_trials")
        self.reset()
        self.create_subscription(Bool, "/kick_trigger", self.cb_kick, 10)
        self.create_subscription(PoseStamped, "/sim/gt/base_pose", lambda m: setattr(self, "gt", m.pose), 10)
        self.create_subscription(PointStamped, "/sim/gt/ball", lambda m: setattr(self, "ball", m.point), 10)
        self.create_subscription(String, "/visual_servo/state", self.cb_servo, 10)
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.cli_start = self.create_client(Trigger, "/sim/start")
        self.cli_init = self.create_client(Trigger, "/sim/set_initialpose")
        self.cli_param = self.create_client(SetParameters, "/visual_servo_kick_node/set_parameters")
        self.rec = None   # --record 면 리스트: (t, kind, data)
        self.create_subscription(String, "/sim/debug/ball_vis", self.cb_vis, 50)
        self.create_subscription(String, "/neck/tracking_state", lambda m: self._rec("neck_state", m.data), 10)
        self.create_subscription(String, "/strategy/phase_dbg", lambda m: None, 1)

    def cb_vis(self, m):
        self._rec("ball_vis", m.data)
        if not self.playing:
            return
        d = json.loads(m.data)
        if d["why"] in ("out_hfov", "out_vfov", "behind") and math.hypot(*d["ball_b"]) < 0.6:
            self.near_blind_frames += 1   # 30 Hz

    def _rec(self, kind, data):
        if self.rec is not None:
            self.rec.append((time.monotonic(), kind, data))

    def reset(self):
        self.gt = self.ball = None
        self.kick_t = None
        self.kick_snap = None
        self.servo_state = None
        self.servo_changes = 0
        self.lost_count = 0
        self.near_blind_frames = 0
        self.playing = False

    def cb_kick(self, m):
        if m.data and self.kick_t is None:
            self.kick_t = time.monotonic()
            self.kick_snap = dict(gt=self.gt, ball=self.ball, est=self.est())

    def cb_servo(self, m):
        self._rec("servo", m.data)
        s = m.data.split()[0] if m.data else m.data
        if s != self.servo_state:
            self.servo_changes += 1
            if s == "LOST" and self.playing:
                self.lost_count += 1
            self.servo_state = s

    def est(self):
        try:
            tr = self.tf.lookup_transform("map", "base_link", Time())
            return (tr.transform.translation.x, tr.transform.translation.y, yaw_q(tr.transform.rotation))
        except Exception:
            return None

    def spin_for(self, sec):
        end = time.monotonic() + sec
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def call(self, cli, req, timeout=5.0):
        if not cli.wait_for_service(timeout_sec=timeout):
            return None
        fut = cli.call_async(req)
        end = time.monotonic() + timeout
        while not fut.done() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
        return fut.result() if fut.done() else None

    def set_params(self, params):
        if not params:
            return True
        req = SetParameters.Request()
        for k, v in params.items():
            pv = ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(v))
            req.parameters.append(ParamMsg(name=k, value=pv))
        res = self.call(self.cli_param, req)
        return res is not None and all(r.successful for r in res.results)


def spawn_list_behind(n, seed):
    """킥오프처럼 공 바로 뒤(골대와 거의 일직선)에서 공을 보고 선 배치."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        r = rng.uniform(0.8, 1.5)
        a = math.radians(rng.uniform(165, 195))
        x, y = r * math.cos(a), r * math.sin(a)
        face = math.degrees(math.atan2(-y, -x)) + rng.uniform(-15, 15)
        out.append((round(x, 2), round(y, 2), round(face, 1)))
    return out


def spawn_list(n, seed):
    """센터 공(0,0) 뒤쪽(우리 진영) 0.8~1.5 m, 공 쪽 ±40° 를 바라보는 배치."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        r = rng.uniform(0.8, 1.5)
        a = math.radians(rng.uniform(120, 240))        # 공 기준 방위 (180 = 정확히 뒤)
        x, y = r * math.cos(a), r * math.sin(a)
        face = math.degrees(math.atan2(-y, -x)) + rng.uniform(-40, 40)
        out.append((round(x, 2), round(y, 2), round(face, 1)))
    return out


def run_trial(w, cfg_name, spawn, logdir, timeout_s):
    env = dict(os.environ)
    env.pop("SIM_SERVO_FIX", None)
    env.update(CONFIGS[cfg_name].get("env", {}))
    procs = []

    def start(cmd, name, **kw):
        f = open(os.path.join(logdir, f"{name}.log"), "w")
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True, env=env, **kw)
        procs.append(p)
        return p

    start([f"{SCRIPTS}/run_bridge.sh", "--scene", SCENE, "--spawn=" + ",".join(map(str, spawn)), "--ball=0,0"], "bridge")
    time.sleep(2)
    start([f"{SCRIPTS}/run_robot.sh", "robo02", env["ROS_DOMAIN_ID"]] + CONFIGS[cfg_name]["launch"], "stack")
    gc = start(["ros2", "run", "humanoid_sim", "sim_gc", "--state", "initial"], "gc", stdin=subprocess.PIPE, text=True)
    w.reset()
    row = dict(config=cfg_name, spawn=spawn, kicked=0, t_kick="", loc_err_m="", loc_err_deg="",
               ball_dir_err_deg="", ball_travel_m="", servo_changes="", lost="", near_blind_s="", note="")
    try:
        w.spin_for(12)
        if w.call(w.cli_start, Trigger.Request()) is None:
            row["note"] = "START 실패"
            return row
        w.spin_for(2)
        w.call(w.cli_init, Trigger.Request())
        w.spin_for(1)
        if not w.set_params(CONFIGS[cfg_name]["servo"]):
            row["note"] = "param 실패"
        gc.stdin.write("set\n"); gc.stdin.flush()
        w.spin_for(1)
        w.reset_kick = None
        w.kick_t = None
        gc.stdin.write("playing\n"); gc.stdin.flush()
        t_play = time.monotonic()
        w.playing = True
        if RECORD:
            w.rec = []
        while time.monotonic() - t_play < timeout_s and w.kick_t is None:
            w.spin_for(0.2)
        if w.kick_t is None:
            row["note"] = f"{timeout_s:.0f} s 안에 킥 없음"
            row["servo_changes"] = w.servo_changes
            return row
        row["kicked"] = 1
        row["t_kick"] = round(w.kick_t - t_play, 1)
        row["servo_changes"] = w.servo_changes
        snap = w.kick_snap
        if snap["gt"] is not None and snap["est"] is not None:
            p = snap["gt"].position
            ex = math.hypot(snap["est"][0] - p.x, snap["est"][1] - p.y)
            ey = math.degrees(math.atan2(math.sin(snap["est"][2] - yaw_q(snap["gt"].orientation)),
                                         math.cos(snap["est"][2] - yaw_q(snap["gt"].orientation))))
            row["loc_err_m"], row["loc_err_deg"] = round(ex, 2), round(ey, 1)
        b0 = np.array([snap["ball"].x, snap["ball"].y]) if snap["ball"] is not None else np.zeros(2)
        w.spin_for(3.5)
        b1 = np.array([w.ball.x, w.ball.y])
        d = b1 - b0
        row["ball_travel_m"] = round(float(np.linalg.norm(d)), 2)
        if np.linalg.norm(d) > 0.3:
            to_goal = GOAL - b0
            ang = math.degrees(math.atan2(d[1], d[0]) - math.atan2(to_goal[1], to_goal[0]))
            row["ball_dir_err_deg"] = round((ang + 180) % 360 - 180, 1)
        return row
    finally:
        row["lost"] = w.lost_count
        row["near_blind_s"] = round(w.near_blind_frames / 30.0, 1)
        w.playing = False
        if RECORD and w.rec:
            t0 = w.rec[0][0]
            with open(os.path.join(logdir, "record.csv"), "w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["t", "kind", "data"])
                for t, k, dd in w.rec:
                    wr.writerow([round(t - t0, 3), k, dd])
            w.rec = None
        for p in procs:
            try:
                os.killpg(p.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        time.sleep(3)
        for p in procs:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="base,lowmin,lowmin_tol")
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=45.0)
    ap.add_argument("--record", action="store_true", help="공 가시성·servo·목 상태 시계열을 판마다 record.csv 로")
    ap.add_argument("--spawn", default=None, help="배치 하나만 지정 x,y,yaw_deg (무작위 대신)")
    ap.add_argument("--behind", action="store_true", help="공 바로 뒤(일직선) 배치만 (킥오프형)")
    ap.add_argument("--out", default=os.path.join(WS, f"logs/kick_trials_{time.strftime('%Y%m%d_%H%M%S')}"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    global RECORD
    RECORD = args.record
    rclpy.init()
    w = Watcher()
    spawns = ([tuple(float(v) for v in args.spawn.split(","))] * args.trials if args.spawn
              else (spawn_list_behind if args.behind else spawn_list)(args.trials, args.seed))
    rows = []
    fields = ["config", "spawn", "kicked", "t_kick", "loc_err_m", "loc_err_deg", "ball_dir_err_deg",
              "ball_travel_m", "servo_changes", "lost", "near_blind_s", "note"]
    with open(os.path.join(args.out, "trials.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for i, sp in enumerate(spawns):
            for cfg in args.configs.split(","):
                d = os.path.join(args.out, f"{i:02d}_{cfg}")
                os.makedirs(d, exist_ok=True)
                row = run_trial(w, cfg, sp, d, args.timeout)
                rows.append(row)
                wr.writerow(row)
                f.flush()
                print(f"[{i + 1}/{len(spawns)}] {cfg:13s} spawn={sp} 분실 {row['lost']}회 근접미검출 {row['near_blind_s']} s → " +
                      (f"킥 {row['t_kick']} s, 측위오차 {row['loc_err_m']} m/{row['loc_err_deg']}°, "
                       f"공방향오차 {row['ball_dir_err_deg']}°, 이동 {row['ball_travel_m']} m"
                       if row["kicked"] else f"실패 ({row['note']})"), flush=True)
    print("\n=== 요약 ===")
    for cfg in args.configs.split(","):
        rs = [r for r in rows if r["config"] == cfg]
        k = [r for r in rs if r["kicked"]]
        tk = [r["t_kick"] for r in k]
        de = [abs(r["ball_dir_err_deg"]) for r in k if r["ball_dir_err_deg"] != ""]
        le = [r["loc_err_m"] for r in k if r["loc_err_m"] != ""]
        lost = sum(r["lost"] for r in rs if r["lost"] != "")
        blind = [r["near_blind_s"] for r in rs if r["near_blind_s"] != ""]
        print(f"{cfg:11s} 킥 {len(k)}/{len(rs)} | 킥까지 중앙 {np.median(tk) if tk else float('nan'):.1f} s "
              f"(최대 {max(tk) if tk else float('nan'):.1f}) | 공방향오차 중앙 {np.median(de) if de else float('nan'):.0f}° | "
              f"킥 순간 측위오차 중앙 {np.median(le) if le else float('nan'):.2f} m | "
              f"분실 합계 {lost}회 | 근접미검출 평균 {np.mean(blind) if blind else float('nan'):.1f} s")
    print(f"결과: {args.out}/trials.csv")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
