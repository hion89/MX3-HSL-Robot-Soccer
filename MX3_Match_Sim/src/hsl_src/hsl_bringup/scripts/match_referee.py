#!/usr/bin/env python3
"""간이 심판 + GameController — 시뮬 경기를 사람 없이 진행한다.

  match_referee.py --team 35 --opp 36 --kickoff 35 --domain 11 [--ready 40] [--half 600]

GC 패킷 (RoboCupGameControlData STRUCT_VERSION 20, 158 B) 을 UDP 3838 로 2 Hz 브로드캐스트한다.
  점수·킥 팀·세트플레이·남은 시간·보조 시간·로봇별 페널티까지 채운다 (실제 GC 와 같은 필드).
진행: INITIAL 3 s → READY → SET 5 s → PLAYING (전반 --half 초) → FINISHED.
브리지(첫 로봇 도메인)의 /sim/event · /sim/ball_touch · /sim/robots 를 듣고 판정한다.
  골         : 득점 → 2 s 뒤 공 센터, 실점 팀 킥오프 (READY 30 s → SET 3 s → PLAYING)
  터치라인 아웃: 마지막에 건드린 팀의 상대에게 킥인 (GC 이름 THROW_IN). 공은 나간 x 의 터치라인 위
  골라인 아웃 : 수비 팀이 마지막 → 공격 팀 코너킥 (코너), 공격 팀이 마지막 → 수비 팀 골킥 (골에어리어 모서리)
  세트플레이는 공이 0.3 m 움직이거나 30 s 지나면 끝 (SET_PLAY_NONE). 그동안 상대는 거리 유지(행동 노드)
  장외 이탈   : 터치/골라인 밖 0.8 m 넘게 나간 로봇 → LEAVING_THE_FIELD 페널티 30 s. 자기 진영 페널티마크
                높이의 터치라인(공 반대쪽)에 세워 두고, 끝나면 그대로 복귀 (측위는 페널티 복귀 가설로 다시 잡는다)
  밀기(Pushing): 상대와 0.6 m 안에서 한쪽만 넘어지면(0.6 s 뒤에도 상대는 서 있음), 넘어지기 0.2~1.0 s 전의
                '상대 쪽으로 다가가는 속도' 를 비교해 서 있는 쪽이 다가간 쪽일 때만 그 로봇에 페널티 30 s +
                넘어진 팀 직접 프리킥. 넘어진 쪽이 다가갔으면(차다가 혼자 넘어짐) 반칙 아님, 둘 다 느리면 우연한 접촉.
                (처음엔 '서 있는 쪽 = 민 쪽' 으로만 봐서 발에 차인 로봇이 퇴장당했다)
  위치 반칙   : SET 이 되는 순간 상대 진영에 있거나, 킥오프 팀이 아닌데 센터서클 안이면 ILLEGAL POSITIONING 30 s
  SET 중 이동 : SET 동안 0.15 m 넘게 움직이면 MOTION IN SET 30 s
  행동 불능   : 넘어진 채 --incapable 초(20) 넘게 못 일어나면 INCAPABLE ROBOT 30 s
  공 정체     : PLAYING 중 공이 --stuck 초 동안 0.2 m 도 안 움직이면 센터 근처로
점수판: /sim/scoreboard (뷰어 위쪽에 표시).
좌표는 월드 기준: --team 팀이 +x 골대를 공격 (장면 생성 규약).
"""
import argparse
import json
import math
import re
import socket
import struct
import time

import rclpy
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String

STATES = {"initial": 0, "ready": 1, "set": 2, "playing": 3, "finished": 4}
STATE_NAMES = {v: k.upper() for k, v in STATES.items()}
SP_NONE, SP_PENALTY_KICK, SP_KICK_IN, SP_GOAL_KICK, SP_CORNER_KICK = 0, 3, 4, 5, 6
SP_NAMES = {0: "", 1: "DIRECT FREE KICK", 2: "INDIRECT FREE KICK", 3: "PENALTY KICK", 4: "KICK-IN",
            5: "GOAL KICK", 6: "CORNER KICK"}
# RoboCupGameControlData.h (HSL, STRUCT_VERSION 20) 의 페널티 코드
PENALTY_ILLEGAL_POSITIONING, PENALTY_MOTION_IN_SET, PENALTY_INCAPABLE_ROBOT = 1, 2, 5
PENALTY_LEAVING_THE_FIELD, PENALTY_PUSHING = 8, 10
PEN_NAMES = {1: "ILLEGAL POSITION", 2: "MOTION IN SET", 5: "INCAPABLE", 8: "LEAVING FIELD", 10: "PUSHING"}
SP_DIRECT_FREE_KICK = 1
MAX_NUM_PLAYERS = 20
# 점수판(ASCII 글꼴)용 로그 영어 치환
EN = {"→": "->", "킥오프 재시작": "kickoff", "골라인 아웃": "goal line out", "터치라인 아웃": "touch line out",
      "마지막": "last", "페널티 해제": "unpenalized", "넘어뜨림": "fell", "넘게 못 일어남": "not up",
      "상대 진영": "opp half", "센터서클": "center circle", "정체": "stuck", "센터 근처로 재배치": "re-placed",
      "에서 복귀": "return", "공 움직임": "ball moved", "시간": "timeout", "이동": "moved", "끝": "over",
      "점수": "score", "팀": "team", "공": "ball"}


def packet(n, state, set_play, kicking, secs, secondary, teams, stopped=False, first_half=True, players=4):
    """teams: [(팀번호, 점수, [(penalty, secs_till), ...]), ...] — game_controller_receiver 의 struct 와 같은 배치."""
    data = struct.pack("<4s10Bhh", b"RGme", 20, n & 0xFF, players, 0, 1 if stopped else 0, 0, state, set_play,
                       1 if first_half else 0, kicking & 0xFF, int(secs), int(secondary))
    for team, score, pens in teams:
        data += struct.pack("<6B2H", team, 0, 1, 1, score, 0, 0, 0)
        for i in range(MAX_NUM_PLAYERS):
            pen, till = pens[i] if i < len(pens) else (0, 0)
            data += struct.pack("<3B", pen, max(0, min(255, int(till))), 0)
    assert len(data) == 158, len(data)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", type=int, default=35, help="+x 골대를 공격하는 팀")
    ap.add_argument("--opp", type=int, default=36)
    ap.add_argument("--kickoff", type=int, default=35)
    ap.add_argument("--domain", type=int, required=True, help="브리지 첫 로봇 도메인")
    ap.add_argument("--ready", type=float, default=40.0)
    ap.add_argument("--ready_after_goal", type=float, default=30.0)
    ap.add_argument("--half", type=float, default=600.0, help="PLAYING 누적 시간 [s] 후 FINISHED")
    ap.add_argument("--set_play_time", type=float, default=30.0)
    ap.add_argument("--penalty_time", type=float, default=30.0)
    ap.add_argument("--leave_margin", type=float, default=0.8, help="라인 밖 이만큼 넘으면 장외 이탈 [m]")
    ap.add_argument("--stuck", type=float, default=45.0)
    ap.add_argument("--no_pushing", action="store_true", help="밀기 반칙 판정 끄기")
    ap.add_argument("--no_positioning", action="store_true", help="SET 위치 반칙 판정 끄기")
    ap.add_argument("--incapable", type=float, default=20.0, help="넘어져 이 시간 넘게 못 일어나면 INCAPABLE [s]")
    ap.add_argument("--kickoff_ball_free", type=float, default=10.0, help="킥오프 후 상대가 공에 들어가지 못하는 시간 [s]")
    ap.add_argument("--hl", type=float, default=4.5)
    ap.add_argument("--hw", type=float, default=3.0)
    ap.add_argument("--penalty_mark", type=float, default=1.5, help="골라인에서 페널티마크까지")
    ap.add_argument("--goal_area", type=str, default="1.0,3.0", help="골에어리어 깊이,폭")
    ap.add_argument("--players", type=int, default=4)
    args = ap.parse_args()
    ga_depth, ga_width = (float(v) for v in args.goal_area.split(","))
    A, B = args.team, args.opp                       # A 가 +x 공격
    hl, hw = args.hl, args.hw

    rclpy.init(domain_id=args.domain)
    node = rclpy.create_node("match_referee")
    pub_ball = node.create_publisher(PointStamped, "/sim/place_ball", 10)
    pub_robot = node.create_publisher(String, "/sim/place_robot", 10)
    pub_board = node.create_publisher(String, "/sim/scoreboard", 10)
    events, ball, touch, robots = [], [0.0, 0.0], {"name": None}, {}
    node.create_subscription(String, "/sim/event", lambda m: events.append(m.data), 10)
    node.create_subscription(PointStamped, "/sim/gt/ball",
                             lambda m: ball.__setitem__(slice(0, 2), [m.point.x, m.point.y]), 10)
    node.create_subscription(String, "/sim/ball_touch", lambda m: touch.__setitem__("name", m.data), 10)
    node.create_subscription(String, "/sim/robots", lambda m: robots.update(json.loads(m.data)), 10)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    st = dict(state=STATES["initial"], kicking=args.kickoff, set_play=SP_NONE, sp_team=0, sp_t=0.0,
              sp_ball=None, stopped=False, ball_free_t=0.0)
    score = {A: 0, B: 0}
    penalties = {}           # 로봇 이름 → (끝나는 시각, 페널티 코드)
    returns = {}             # 로봇 이름 → 복귀 위치 (페널티 끝날 때 사람이 다시 세우는 자리)
    upright = {}             # 로봇 이름 → 서 있는가 (직전)
    fall_checks = []         # (판정 시각, 넘어진 로봇, [(상대, 상대 접근속도, 넘어진 쪽 접근속도)])
    hist = {}                # 로봇 이름 → [(시각, x, y)] 최근 2 s (접근 속도 계산용)
    fallen_since = {}        # 로봇 이름 → 넘어진 시각
    set_pose = {}            # SET 진입 시 위치
    plan = [("initial", 3.0), ("ready", args.ready), ("set", 5.0), ("playing", None)]
    plan_t = time.monotonic()
    pending = None
    playing_time = 0.0
    last_send = last_board = 0.0
    n = 0
    last_move = (time.monotonic(), tuple(ball))
    last_event = ""

    def log(msg):
        nonlocal last_event
        last_event = msg
        print(f"[referee {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def team_of(name):
        m = re.match(r"t(\d+)p(\d+)", name or "")
        return (int(m.group(1)), int(m.group(2))) if m else (None, None)

    def place(x, y):
        m = PointStamped()
        m.header.frame_id = "map"
        m.point.x, m.point.y = float(x), float(y)
        pub_ball.publish(m)

    def start_set_play(kind, team, x, y):
        place(x, y)
        st.update(set_play=kind, sp_team=team, kicking=team, sp_t=time.monotonic(), sp_ball=(x, y), stopped=False)
        log(f"{SP_NAMES[kind]} → 팀 {team}, 공 ({x:+.2f}, {y:+.2f})")

    def restart_kickoff(kicking):
        nonlocal plan, plan_t
        place(0.0, 0.0)
        st.update(kicking=kicking, set_play=SP_NONE, stopped=False)
        plan = [("ready", args.ready_after_goal), ("set", 3.0), ("playing", None)]
        plan_t = time.monotonic()
        log(f"킥오프 재시작 → 팀 {kicking}")

    def team_pens(team):
        now = time.monotonic()
        out = [(0, 0)] * args.players
        for name, (until, code) in penalties.items():
            t, p = team_of(name)
            if t == team and 1 <= p <= args.players:
                out[p - 1] = (code, until - now)
        return out

    def penalize(name, code, why):
        """페널티: 자기 진영 페널티마크 높이 터치라인(공 반대쪽)에 세워 두고 penalty_time 뒤 복귀."""
        team, _ = team_of(name)
        own = -1.0 if team == A else 1.0
        side = -1.0 if ball[1] > 0 else 1.0
        busy = sum(1 for k in penalties if team_of(k)[0] == team)   # 같은 팀 동시 페널티는 0.6 m 씩 비켜 세움
        px, py = own * (hl - args.penalty_mark - 0.6 * busy), side * (hw + 0.15)
        pub_robot.publish(String(data=f"{name} {px:.2f} {py:.2f} {-90.0 * side:.0f}"))
        penalties[name] = (time.monotonic() + args.penalty_time, code)
        returns[name] = f"{name} {px:.2f} {py:.2f} {-90.0 * side:.0f}"
        log(f"{name} {PEN_NAMES[code]} ({why}) → {args.penalty_time:.0f} s, ({px:+.1f}, {py:+.1f}) 에서 복귀")

    def secondary_timer():
        """GC 보조 시간 (실제 GC 의 secondaryTime 처럼): (이름, 남은 초) 또는 None.
        READY·SET·INITIAL 남은 시간 > 세트플레이 남은 시간 > 킥오프 뒤 공 인플레이까지."""
        now_ = time.monotonic()
        if plan and plan[0][1] is not None and STATES[plan[0][0]] == st["state"]:
            return STATE_NAMES[st["state"]], max(0.0, plan[0][1] - (now_ - plan_t))
        if st["set_play"]:
            return SP_NAMES[st["set_play"]], max(0.0, args.set_play_time - (now_ - st["sp_t"]))
        if st["state"] == STATES["playing"] and now_ < st["ball_free_t"]:
            return "BALL FREE", st["ball_free_t"] - now_
        return None

    def board():
        """점수판 JSON — 브리지가 뷰어 왼쪽 위(score) / 오른쪽 위(gc) 상자로 그린다. 뷰어 글꼴은 ASCII 만."""
        mm, ss = divmod(int(max(0.0, args.half - playing_time)), 60)
        sp = SP_NAMES.get(st["set_play"], "")
        sec = secondary_timer()
        gc = [["STATE", STATE_NAMES[st["state"]]], ["TIME", f"{mm:02d}:{ss:02d}"],
              ["TIMER", f"{sec[0]} {int(math.ceil(sec[1]))} s" if sec else "-"],
              ["KICKING", str(st["kicking"])], ["SET PLAY", f"{sp} ({st['sp_team']})" if sp else "-"]]
        for k, (u, c) in sorted(penalties.items()):
            gc.append([k, f"{PEN_NAMES[c]} {max(0, int(u - time.monotonic()))} s"])
        evt = last_event
        for ko, en in EN.items():
            evt = evt.replace(ko, en)
        evt = re.sub(r"\s+", " ", re.sub(r"[^\x20-\x7e]", "", evt)).strip()
        if evt:
            gc.append(["LAST", evt[:48]])
        score_box = [[f"{A} (BLUE)", str(score[A])], [f"{B} (RED)", str(score[B])]]
        return json.dumps({"score": score_box, "gc": gc})

    t_prev = time.monotonic()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            now = time.monotonic()
            # ── 진행표 ──
            if plan:
                name, dur = plan[0]
                if STATES[name] != st["state"]:
                    if name == "set" and not args.no_positioning:
                        for rn, (x, y, _yaw, _up) in list(robots.items()):
                            team, _ = team_of(rn)
                            if team is None or rn in penalties:
                                continue
                            own = -1.0 if team == A else 1.0
                            wrong_half = x * own < -0.1                 # 상대 진영으로 0.1 m 넘게
                            in_circle = team != st["kicking"] and math.hypot(x, y) < 0.75 - 0.05
                            if wrong_half or in_circle:
                                penalize(rn, PENALTY_ILLEGAL_POSITIONING,
                                         "상대 진영" if wrong_half else "센터서클")
                    if name == "set":
                        set_pose.clear()
                        set_pose.update({rn: (v[0], v[1]) for rn, v in robots.items()})
                    if name == "playing" and st["state"] == STATES["set"]:
                        st["ball_free_t"] = now + args.kickoff_ball_free     # 킥오프: 상대는 10 s 또는 공이 움직일 때까지 대기
                    st["state"] = STATES[name]
                    last_move = (now, tuple(ball))
                    log(f"state → {name.upper()}" + (f" ({dur:.0f} s)" if dur else ""))
                if dur is not None and now - plan_t >= dur:
                    plan.pop(0)
                    plan_t = now
                elif dur is None:
                    plan = []
            if st["state"] == STATES["playing"]:
                playing_time += now - t_prev
                if playing_time >= args.half:
                    st.update(state=STATES["finished"], set_play=SP_NONE)
                    log(f"FINISHED — 점수 {A} {score[A]} : {score[B]} {B}")
            t_prev = now
            playing = st["state"] == STATES["playing"]
            # ── 브리지 이벤트 (골·아웃) ──
            while events:
                ev = events.pop(0)
                if not playing:
                    continue
                m = re.search(r"last=(\S+)", ev)
                last_team, _ = team_of(m.group(1) if m else touch["name"])
                bx, by = ball
                if ev.startswith("GOAL_US"):
                    score[A] += 1
                    log(f"GOAL {A}!  {A} {score[A]} : {score[B]} {B}")
                    st.update(set_play=SP_NONE, stopped=True)
                    pending = (now + 2.0, lambda: restart_kickoff(B))
                elif ev.startswith("GOAL_THEM"):
                    score[B] += 1
                    log(f"GOAL {B}!  {A} {score[A]} : {score[B]} {B}")
                    st.update(set_play=SP_NONE, stopped=True)
                    pending = (now + 2.0, lambda: restart_kickoff(A))
                elif ev.startswith("OUT_TOUCH"):
                    team = B if last_team == A else A if last_team == B else (A if bx < 0 else B)
                    x = max(-hl + 0.5, min(hl - 0.5, bx))
                    y = math.copysign(hw - 0.05, by)
                    log(f"터치라인 아웃 (마지막 {last_team or '?'})")
                    st["stopped"] = True
                    pending = (now + 2.0, lambda t=team, x=x, y=y: start_set_play(SP_KICK_IN, t, x, y))
                elif ev.startswith("OUT_GOAL"):
                    defender = B if bx > 0 else A            # +x 골라인은 B 가 지킨다
                    attacker = A if defender == B else B
                    sy = math.copysign(1.0, by if abs(by) > 0.05 else 1.0)
                    sx = math.copysign(1.0, bx)
                    if last_team == defender:
                        kind, team, x, y = SP_CORNER_KICK, attacker, sx * (hl - 0.05), sy * (hw - 0.05)
                    else:
                        kind, team, x, y = SP_GOAL_KICK, defender, sx * (hl - ga_depth), sy * ga_width / 2
                    log(f"골라인 아웃 (마지막 {last_team or '?'})")
                    st["stopped"] = True
                    pending = (now + 2.0, lambda k=kind, t=team, x=x, y=y: start_set_play(k, t, x, y))
            if pending and now >= pending[0]:
                fn = pending[1]
                pending = None
                fn()
                last_move = (now, tuple(ball))
            # ── 세트플레이 끝 ──
            if st["set_play"] != SP_NONE and st["sp_ball"] is not None:
                moved = math.hypot(ball[0] - st["sp_ball"][0], ball[1] - st["sp_ball"][1]) > 0.3
                if (moved and now - st["sp_t"] > 0.5) or now - st["sp_t"] > args.set_play_time:
                    log(f"{SP_NAMES[st['set_play']]} 끝 ({'공 움직임' if moved else '시간'})")
                    st.update(set_play=SP_NONE, sp_ball=None)
            # ── 페널티 해제 ──
            for name, (until, _code) in list(penalties.items()):
                if now >= until:
                    del penalties[name]
                    if name in returns:           # 사람이 터치라인에 필드를 보게 다시 세운다 (실제 복귀 절차)
                        pub_robot.publish(String(data=returns.pop(name)))
                    log(f"{name} 페널티 해제")
            # ── 장외 이탈 ──
            if playing:
                for name, (x, y, _yaw, _up) in list(robots.items()):
                    if name not in penalties and (abs(x) > hl + args.leave_margin or abs(y) > hw + args.leave_margin):
                        penalize(name, PENALTY_LEAVING_THE_FIELD, f"({x:+.1f}, {y:+.1f})")
            # ── 밀기: 상대와 붙어 있다 한쪽만 넘어짐 → 누가 다가갔나로 판정 ──
            for name, (x, y, _yaw, _up) in robots.items():
                h = hist.setdefault(name, [])
                if not h or now - h[-1][0] > 0.15:
                    h.append((now, x, y))
                    while h and now - h[0][0] > 2.0:
                        h.pop(0)

            def vel(name, t_end):
                """t_end−0.8 ~ t_end 사이 평균 속도 (넘어지는 동작 자체는 빼려고 넘어지기 0.2 s 전까지)."""
                h = [p for p in hist.get(name, []) if t_end - 0.8 <= p[0] <= t_end]
                if len(h) < 2 or h[-1][0] - h[0][0] < 0.3:
                    return 0.0, 0.0
                dt_ = h[-1][0] - h[0][0]
                return (h[-1][1] - h[0][1]) / dt_, (h[-1][2] - h[0][2]) / dt_

            for name, (x, y, _yaw, up) in list(robots.items()):
                was = upright.get(name, True)
                upright[name] = up
                if playing and not args.no_pushing and was and not up and name not in penalties:
                    team, _ = team_of(name)
                    vvx, vvy = vel(name, now - 0.2)
                    cands = []
                    for o, (ox, oy, _oyaw, oup) in robots.items():
                        if team_of(o)[0] in (None, team) or not oup or o in penalties:
                            continue
                        dd = math.hypot(x - ox, y - oy)
                        if dd >= 0.6:
                            continue
                        ux, uy = (x - ox) / max(dd, 1e-6), (y - oy) / max(dd, 1e-6)   # 상대 → 넘어진 로봇
                        ovx, ovy = vel(o, now - 0.2)
                        cands.append((dd, o, ovx * ux + ovy * uy, -(vvx * ux + vvy * uy)))
                    if cands:
                        cands.sort()
                        fall_checks.append((now + 0.6, name, [(o, ao, av) for _d, o, ao, av in cands]))
            for chk in list(fall_checks):
                if now < chk[0]:
                    continue
                fall_checks.remove(chk)
                _t, victim, cands = chk
                if not playing or victim in penalties:
                    continue
                pushed = False
                for o, a_o, a_v in cands:
                    if not robots.get(o, (0, 0, 0, False))[3] or o in penalties:
                        continue                        # 상대도 넘어졌으면 서로 부딪힌 것
                    if a_o > 0.08 and a_o > a_v + 0.05:
                        penalize(o, PENALTY_PUSHING, f"{victim} 넘어뜨림, 접근 {a_o:.2f} vs {a_v:.2f} m/s")
                        vteam, _ = team_of(victim)
                        st.update(set_play=SP_DIRECT_FREE_KICK, sp_team=vteam, kicking=vteam, sp_t=now,
                                  sp_ball=tuple(ball), stopped=False)
                        log(f"DIRECT FREE KICK → 팀 {vteam} (공 그 자리)")
                        pushed = True
                        break
                if not pushed:
                    o, a_o, a_v = cands[0]
                    log(f"{victim} 접촉 넘어짐 — 반칙 아님 ({o} 접근 {a_o:.2f}, {victim} 접근 {a_v:.2f} m/s)")
            # ── 행동 불능: 넘어진 채 못 일어남 ──
            for name, (x, y, _yaw, up) in list(robots.items()):
                if up or name in penalties:
                    fallen_since.pop(name, None)
                    continue
                fallen_since.setdefault(name, now)
                if playing and now - fallen_since[name] > args.incapable:
                    fallen_since.pop(name, None)
                    penalize(name, PENALTY_INCAPABLE_ROBOT, f"{args.incapable:.0f} s 넘게 못 일어남")
            # ── SET 중 이동 ──
            if st["state"] == STATES["set"]:
                for name, (x, y, _yaw, _up) in list(robots.items()):
                    p0 = set_pose.get(name)
                    if p0 and name not in penalties and math.hypot(x - p0[0], y - p0[1]) > 0.15:
                        penalize(name, PENALTY_MOTION_IN_SET, f"{math.hypot(x - p0[0], y - p0[1]):.2f} m 이동")
            # ── 공 정체 ──
            if math.hypot(ball[0] - last_move[1][0], ball[1] - last_move[1][1]) > 0.2:
                last_move = (now, tuple(ball))
            elif playing and now - last_move[0] > args.stuck:
                log(f"공 {args.stuck:.0f} s 정체 → 센터 근처로 재배치")
                place(0.0, 0.5 if ball[1] < 0 else -0.5)
                last_move = (now, tuple(ball))
            # ── GC 패킷 · 점수판 ──
            if now - last_send >= 0.5:
                last_send = now
                secs = max(0.0, args.half - playing_time)
                sec = secondary_timer()
                secondary = sec[1] if sec else 0
                data = packet(n, st["state"], st["set_play"], st["kicking"], secs, secondary,
                              [(A, score[A], team_pens(A)), (B, score[B], team_pens(B))],
                              stopped=st["stopped"], players=args.players)
                sock.sendto(data, ("255.255.255.255", 3838))
                n += 1
            if now - last_board >= 0.5:
                last_board = now
                pub_board.publish(String(data=board()))
    except KeyboardInterrupt:
        pass
    log(f"종료 — 점수 {A} {score[A]} : {score[B]} {B}")


if __name__ == "__main__":
    main()
