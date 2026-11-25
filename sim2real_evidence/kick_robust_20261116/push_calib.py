"""킥 디딤발 쪽 외력 크기 보정: 힘을 단계별로 주며 넘어짐·몸 roll 을 본다 (MuJoCo, 로봇·ROS 무관).

외력: 한쪽 발이 떠 있고 그 다리 골반 roll 벌림이 40° 를 넘는 순간부터 push_s 동안,
몸통에 디딤발 방향 수평 힘 (mujoco_kick_stress.KickPush).
  python push_calib.py --policy a.onnx --trials 20 --out results_push_calib
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os

import numpy as np

import kick_real_compare as kc
import mujoco_kick_stress as ks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", action="append", required=True)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--force", type=float, action="append")
    ap.add_argument("--push-s", type=float, default=0.5)
    ap.add_argument("--out", default="results_push_calib")
    args = ap.parse_args()
    import onnxruntime as ort
    forces = args.force or [0, 5, 10, 15, 20, 25, 30, 40]
    sim = ks.Sim(ks.DEFAULT_SCENE)
    real = dataclasses.replace(ks.Cond(), name="real_like", volt=10.3, imu_delay_ms=10.0, hold_s=0.6,
                               cmd_vx=0.20, filter_tau=0.02, mu=0.6, mu_torsion=0.1)
    models = [("sim", {}), ("mass6.5_kp30", dict(base_add_kg=0.8, kp=30.0))]
    os.makedirs(args.out, exist_ok=True)
    out = []
    for pol in args.policy:
        sess = ort.InferenceSession(pol, providers=["CPUExecutionProvider"])
        for mname, mod in models:
            kc.apply_model(sim, **mod)
            for F in forces:
                c = dataclasses.replace(real, push_N=F, push_s=args.push_s)
                res = []
                for i in range(args.trials):
                    rows, fell_t = kc.run(sim, sess, c, 1000 + i, 2.0, 1.5, 0.03)
                    s = kc.summarize(rows, fell_t)
                    res.append(s)
                agg = {k: float(np.mean([r[k] for r in res])) for k in res[0] if k != "kick_leg"}
                agg.update(policy=os.path.basename(pol), model=mname, force=F, push_s=args.push_s)
                out.append(agg)
                print(f"{os.path.basename(pol):26s} {mname:13s} F {F:4.0f} N: fall {100*agg['fell']:3.0f}% | pushed {100*agg['pushed']:3.0f}% | "
                      f"roll max {agg['roll_max']:4.0f} | kick hip roll {agg['kick_hip_roll_max']:3.0f} | vball {agg['vball_max']:.2f}", flush=True)
    kc.apply_model(sim)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
