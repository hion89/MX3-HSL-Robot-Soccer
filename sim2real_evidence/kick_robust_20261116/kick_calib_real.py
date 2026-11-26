"""실물 킥 1회차 (2026-11-26 kick_push_31800, 넘어짐) 에 MuJoCo 모델을 맞춘다.

실물: 세기 0.5, 공 0.30 m 정면, 배터리 12.0 V, 필터 0.03, hold 0.6, cmd_vx 0.2.
  1.9 s 에 넘어짐 (PROTECT). 그 직전 디딤발 골반 pitch·발목 pitch·골반 roll PWM 100 %, 전류 6 A, 전압 10.3 V.
  차는 다리 골반 roll 실제 +40°. 11-25 에는 22300·27500·30899 도 13 회 중 11 회 넘어졌다.
MuJoCo 는 같은 조건에서 거의 넘어지지 않는다 → 디딤발 부하가 큰 쪽으로 모델 손잡이를 바꿔 가며
"정책 셋 다 1.5~2.5 s 에 대부분 넘어짐" 이 되는 조건을 찾는다.

  python kick_calib_real.py --policy A.onnx --policy B.onnx --trials 12 --out results_calib_real_20261126
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import kick_real_compare as kc
import mujoco_kick_stress as ks

SUP = {  # 차는 다리가 R 이면 디딤발 L 의 (골반 pitch, 골반 roll, 발목 pitch) 인덱스
    "R": [ks.POLICY_JOINTS.index(n) for n in ("l_pelvis_pitch_joint", "l_pelvis_roll_joint", "l_ankle_pitch_joint")],
    "L": [ks.POLICY_JOINTS.index(n) for n in ("r_pelvis_pitch_joint", "r_pelvis_roll_joint", "r_ankle_pitch_joint")],
}


# 게이트용 고정 조건 (2026-11-26 보정 결과). motor_t1 이 실물 1회차와 가장 잘 맞는다 (L3 92 %, 30899·27500 100 %, 1.8~2.2 s).
GATE = [
    dict(add_kg=0.8, cap=3.0, kp=50.0, volt=10.3, stall_k=0.7),   # motor_t1
    dict(add_kg=0.8, cap=3.5, kp=50.0, volt=11.1, stall_k=0.85),  # 중간
    dict(add_kg=1.6, cap=4.0, kp=50.0, volt=12.0, stall_k=1.0),   # 무거운 몸 + 4 N·m
]


def grid():
    out = []
    for add_kg, cap, kp, volt, stall_k in itertools.product((0.8, 1.6, 2.4), (5.0, 4.0, 3.0), (50.0, 30.0),
                                                            (12.0, 10.3), (1.0, 0.7)):
        out.append(dict(add_kg=add_kg, cap=cap, kp=kp, volt=volt, stall_k=stall_k))
    return out


_SIM = {}


def work(job):
    pol, mdl, seeds = job
    import onnxruntime as ort
    if "sim" not in _SIM:
        _SIM["sim"] = ks.Sim(ks.DEFAULT_SCENE)
        _SIM["sess"] = {}
    sim = _SIM["sim"]
    if pol not in _SIM["sess"]:
        _SIM["sess"][pol] = ort.InferenceSession(pol, providers=["CPUExecutionProvider"])
    sess = _SIM["sess"][pol]
    kc.apply_model(sim, base_add_kg=mdl["add_kg"], kp=mdl["kp"], cap=mdl["cap"])
    ks.STALL_12V = 7.3 * mdl["stall_k"]
    c = dataclasses.replace(ks.Cond(), name="real_t1", volt=mdl["volt"], imu_delay_ms=10.0, hold_s=0.6, cmd_vx=0.20,
                            filter_tau=0.03, mu=0.6, mu_torsion=0.1, strength=0.5)
    res = []
    for sd in seeds:
        rows, fell_t = kc.run(sim, sess, c, sd, 2.0, 1.5, 0.03, ball_x=0.30)
        s = kc.summarize(rows, fell_t)
        sat = np.abs(np.array([r["sat"] for r in rows]))[:, SUP[s["kick_leg"]]]
        res.append(dict(fell=s["fell"], fell_t=s["fell_t"], kick_hip_roll=s["kick_hip_roll_max"],
                        roll_max=s["roll_max"], sup_sat95=float((sat.max(1) >= 0.95).mean()), vball=s["vball_max"]))
    return pol, mdl, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", action="append", required=True)
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--out", default="results_calib_real_20261126")
    ap.add_argument("--preset", default="grid", choices=["grid", "gate"], help="gate = 보정된 3 조건만 (조기 종료 게이트)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    models = GATE if args.preset == "gate" else grid()
    jobs = [(p, m, list(range(2000, 2000 + args.trials))) for m in models for p in args.policy]
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (pol, mdl, res) in enumerate(ex.map(work, jobs)):
            f = np.array([r["fell"] for r in res])
            ft = [r["fell_t"] for r in res if r["fell"]]
            row = dict(policy=os.path.basename(pol), **mdl, fall=float(f.mean()),
                       fall_t=float(np.median(ft)) if ft else -1.0,
                       kick_hip_roll=float(np.mean([r["kick_hip_roll"] for r in res])),
                       sup_sat95=float(np.mean([r["sup_sat95"] for r in res])),
                       vball=float(np.mean([r["vball"] for r in res])))
            rows.append(row)
            print("%-26s kg+%.1f cap%.0f kp%.0f V%.1f stall×%.1f: fall %3.0f%% t %.2f | kick roll %4.0f | sup sat %.2f | v %.2f"
                  % (row["policy"][:26], mdl["add_kg"], mdl["cap"], mdl["kp"], mdl["volt"], mdl["stall_k"],
                     100 * row["fall"], row["fall_t"], row["kick_hip_roll"], row["sup_sat95"], row["vball"]), flush=True)
    json.dump(rows, open(os.path.join(args.out, "calib.json"), "w"), indent=1)
    for pol in args.policy:
        rs = [r for r in rows if r["policy"] == os.path.basename(pol)]
        print("- %s: motor fall mean %.1f %% (%s), kick roll %.0f, v %.2f" % (
            os.path.basename(pol), 100 * np.mean([r["fall"] for r in rs]),
            " / ".join("%.0f" % (100 * r["fall"]) for r in rs),
            np.mean([r["kick_hip_roll"] for r in rs]), np.mean([r["vball"] for r in rs])))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
