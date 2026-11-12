#!/usr/bin/env python3
"""간이 GameController — 실제 GC 와 같은 RoboCupGameControlData(STRUCT_VERSION 20, 158 B)를
UDP 3838 로 브로드캐스트한다. 형식은 game_controller_receiver_node.py 의 struct 정의를 그대로 따른다.

  python3 sim_gc.py --team 35 --opp 36 --state playing
실행 중 표준입력으로 상태를 바꾼다:  initial / ready / set / playing / finished / kickoff <팀번호> / quit
"""
import argparse
import socket
import struct
import sys
import threading
import time

_CONTROL_FMT = "<4s10Bhh"
_TEAM_HEAD_FMT = "<6B2H"
_ROBOT_FMT = "<3B"
MAX_NUM_PLAYERS = 20
STATES = {"initial": 0, "ready": 1, "set": 2, "playing": 3, "finished": 4}


def packet(n, state, kicking, teams, secs, first_half=True, players=4):
    data = struct.pack(_CONTROL_FMT, b"RGme", 20, n & 0xFF, players, 0, 0, 0, state, 0,
                       1 if first_half else 0, kicking, secs, 0)
    for team in teams:
        data += struct.pack(_TEAM_HEAD_FMT, team, 0, 1, 1, 0, 0, 0, 0)
        data += b"".join(struct.pack(_ROBOT_FMT, 0, 0, 0) for _ in range(MAX_NUM_PLAYERS))
    assert len(data) == 158, len(data)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", type=int, default=35)
    ap.add_argument("--opp", type=int, default=36)
    ap.add_argument("--state", default="playing", choices=list(STATES))
    ap.add_argument("--kickoff", type=int, default=None, help="킥오프 팀 (기본 우리 팀)")
    ap.add_argument("--addr", default="255.255.255.255")
    ap.add_argument("--port", type=int, default=3838)
    ap.add_argument("--hz", type=float, default=2.0)
    ap.add_argument("--sequence", default=None,
                    help="자동 진행, 예: initial:5,ready:45,set:5,playing  (마지막 상태는 유지)")
    args = ap.parse_args()

    st = dict(state=STATES[args.state], kicking=args.kickoff or args.team, start=time.monotonic())
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    def loop():
        n = 0
        while True:
            secs = max(0, 600 - int(time.monotonic() - st["start"]))
            sock.sendto(packet(n, st["state"], st["kicking"], (args.team, args.opp), secs),
                        (args.addr, args.port))
            n += 1
            time.sleep(1.0 / args.hz)

    threading.Thread(target=loop, daemon=True).start()
    print(f"[sim_gc] {args.addr}:{args.port} team {args.team} vs {args.opp}, state={args.state}", flush=True)
    if args.sequence:
        for item in args.sequence.split(","):
            name, _, dur = item.partition(":")
            st["state"] = STATES[name]
            print(f"[sim_gc] state → {name}" + (f" ({dur} s)" if dur else ""), flush=True)
            if not dur:
                break
            time.sleep(float(dur))
    for line in sys.stdin:
        cmd = line.strip().split()
        if not cmd:
            continue
        if cmd[0] == "quit":
            break
        if cmd[0] in STATES:
            st["state"] = STATES[cmd[0]]
            print(f"[sim_gc] state → {cmd[0]}")
        elif cmd[0] == "kickoff" and len(cmd) > 1:
            st["kicking"] = int(cmd[1])
            print(f"[sim_gc] kickoff → {cmd[1]}")
        else:
            print("명령: initial / ready / set / playing / finished / kickoff <팀> / quit")


if __name__ == "__main__":
    main()
