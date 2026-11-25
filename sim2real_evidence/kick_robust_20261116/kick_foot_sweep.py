"""발바닥 접촉 형상을 바꿔 실물 킥 낙상(디딤발 모서리 들림)이 재현되는지 본다 (MuJoCo, 로봇·ROS 무관).

실물 발: 17 x 8 cm, 각지지 않은(둥근) 모서리, 돌기 있는 3D 프린트 고무 패드, 인조잔디.
MuJoCo 기본 발: 메시 볼록 껍질 (17 x 8 cm, 날카로운 모서리, 단단함).

발 형상 (ankle_roll_link 몸체 좌표: x 앞 -0.028~0.142, y ±0.04, 바닥 z = -0.04):
  mesh        : 원래 메시 접촉
  box{W}      : 폭 W cm 상자 (모서리가 눌려 실효 지지폭이 줄어든 경우)
  caps{R}     : 길이 방향 캡슐 2개로 둥근 옆 모서리 (반지름 R cm, 바깥 폭 8 cm 유지)
solref_tc 로 패드·잔디 부드러움을 같이 바꾼다.

  python kick_foot_sweep.py --policy a.onnx --trials 10 --out results_foot_sweep
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os

import mujoco
import numpy as np

import kick_real_compare as kc
import mujoco_kick_stress as ks

FOOT_BODIES = ("l_ankle_roll_link", "r_ankle_roll_link")
X0, X1, ZB = -0.028, 0.142, -0.04


def build_model(foot: str):
    spec = mujoco.MjSpec.from_file(ks.DEFAULT_SCENE)
    if foot == "mesh":
        return spec.compile()
    for bn in FOOT_BODIES:
        body = spec.body(ks.PREFIX + bn)
        for g in body.geoms:
            if g.contype or g.conaffinity:
                cls_fr = g.friction
                g.contype, g.conaffinity = 0, 0
        xc, hx = (X0 + X1) / 2, (X1 - X0) / 2
        if foot.startswith("box"):
            w = float(foot[3:]) / 100.0
            g = body.add_geom()
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = [hx, w / 2, 0.004]
            g.pos = [xc, 0.0, ZB + 0.004]
        elif foot.startswith("caps"):
            r = float(foot[4:]) / 100.0
            for y in (0.04 - r, -(0.04 - r)):
                g = body.add_geom()
                g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
                g.size = [r, hx - r, 0]
                g.pos = [xc, y, ZB + r]
                g.quat = [0.7071068, 0.0, 0.7071068, 0.0]   # 캡슐 축(z)을 x 로
        else:
            raise ValueError(foot)
        for g in body.geoms:
            if g.contype == 0 and g.conaffinity == 0 and g.type == mujoco.mjtGeom.mjGEOM_MESH:
                continue
            g.contype, g.conaffinity = 1, 1
            g.condim = 4
            g.mass = 0.0
            g.density = 0.0
    return spec.compile()


def make_sim(foot: str):
    m = build_model(foot)
    orig = mujoco.MjModel.from_xml_path
    mujoco.MjModel.from_xml_path = staticmethod(lambda _p: m)
    try:
        sim = ks.Sim(ks.DEFAULT_SCENE)
    finally:
        mujoco.MjModel.from_xml_path = orig
    return sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", action="append", required=True)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--foot", action="append")
    ap.add_argument("--solref", type=float, action="append")
    ap.add_argument("--out", default="results_foot_sweep")
    args = ap.parse_args()
    import onnxruntime as ort
    feet = args.foot or ["mesh", "box8", "box6", "box5", "box4", "caps1.5", "caps2.5"]
    solrefs = args.solref or [0.02, 0.06]
    base = ks.Cond()
    real = dataclasses.replace(base, name="real_like", volt=10.3, imu_delay_ms=10.0, hold_s=0.6,
                               cmd_vx=0.20, filter_tau=0.02, mu=0.6, mu_torsion=0.1)
    mod = dict(base_add_kg=0.8, kp=30.0)   # 실물 질량 6.5 kg, P800 강성
    os.makedirs(args.out, exist_ok=True)
    summary = []
    sess = {p: ort.InferenceSession(p, providers=["CPUExecutionProvider"]) for p in args.policy}
    for foot in feet:
        sim = make_sim(foot)
        kc._DEF.clear()
        kc.apply_model(sim, **mod)
        for tc in solrefs:
            c = dataclasses.replace(real, solref_tc=tc)
            for pol, s in sess.items():
                res = []
                for i in range(args.trials):
                    rows, fell_t = kc.run(sim, s, c, 1000 + i, 2.0, 1.5, 0.03)
                    res.append(kc.summarize(rows, fell_t))
                agg = {k: float(np.mean([r[k] for r in res])) for k in res[0] if k != "kick_leg"}
                agg.update(policy=os.path.basename(pol), foot=foot, solref=tc)
                summary.append(agg)
                print(f"{os.path.basename(pol):26s} foot {foot:8s} solref {tc:<5g}: fall {100*agg['fell']:3.0f}% | "
                      f"roll max {agg['roll_max']:4.0f} | kick hip roll {agg['kick_hip_roll_max']:3.0f} | "
                      f"sup err {agg['support_hip_roll_err_max']:3.0f} | vball {agg['vball_max']:.2f}", flush=True)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
