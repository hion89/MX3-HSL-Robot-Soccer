"""Rank kick checkpoints with the MuJoCo sim2sim gate (Windows PC or server, no Isaac / ROS).

  python gate_checkpoints.py --ckpt-dir <run folder copied from server> --every 500 --trials 300
  python gate_checkpoints.py --ckpt a/model_22800.pt --ckpt b/model_23300.pt --baseline ../../XM_Kick_Policy/policy/policy.onnx

Steps: model_*.pt -> ONNX (same layout as scripts/kick_v1_5/export_onnx.py: obs[1,265] -> actions[1,12], ELU)
-> mujoco_kick_stress.py --suite gate -> ranking.csv / ranking.md.
Ranking key: worst-case fall % over the gate cells, then mean fall %. Candidates whose kick success < 95 % or whose
nominal ball speed drops below 85 % of the baseline are flagged (a policy that stops kicking is not an improvement).
Simulation ranking only; the real-robot check is separate.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE_ONNX = os.path.join(REPO, "XM_Kick_Policy", "policy", "policy.onnx")


def export(ckpt: str, out: str):
    import torch
    import torch.nn as nn
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
    lin = sorted({int(k.split(".")[1]) for k in sd if k.startswith("actor.") and k.endswith(".weight")})
    layers = []
    for i, li in enumerate(lin):
        w = sd[f"actor.{li}.weight"]
        m = nn.Linear(w.shape[1], w.shape[0])
        m.weight.data.copy_(w)
        m.bias.data.copy_(sd[f"actor.{li}.bias"])
        layers.append(m)
        if i < len(lin) - 1:
            layers.append(nn.ELU())
    net = nn.Sequential(*layers).eval()
    if layers[0].in_features != 265 or layers[-1].out_features != 12:
        raise SystemExit(f"{ckpt}: actor {layers[0].in_features}->{layers[-1].out_features}, expected 265->12")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.onnx.export(net, torch.zeros(1, 265), out, input_names=["obs"], output_names=["actions"],
                      opset_version=18, dynamo=False)


def main():
    if hasattr(sys.stdout, "reconfigure"):  # Windows cp949 콘솔에서 ⚠ 출력 시 UnicodeEncodeError 방지
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", action="append", default=[])
    ap.add_argument("--ckpt-dir", action="append", default=[])
    ap.add_argument("--every", type=int, default=500, help="keep iterations divisible by this (ckpt-dir)")
    ap.add_argument("--min-iter", type=int, default=0)
    ap.add_argument("--baseline", default=BASE_ONNX, help="reference ONNX (22300) evaluated in the same run")
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=os.path.join(HERE, "gate_out"))
    ap.add_argument("--suite", default="gate", choices=["gate", "push"], help="push = 킥 디딤발 쪽 외력 4~12 N")
    args = ap.parse_args()

    ckpts = list(args.ckpt)
    for d in args.ckpt_dir:
        for p in glob.glob(os.path.join(d, "model_*.pt")):
            it = int(re.findall(r"model_(\d+)\.pt", p)[0])
            if it >= args.min_iter and it % args.every == 0:
                ckpts.append(p)
    ckpts = sorted(set(ckpts), key=lambda p: (os.path.dirname(p), int(re.findall(r"(\d+)", os.path.basename(p))[-1])))
    if not ckpts:
        raise SystemExit("no checkpoints selected")
    onnx_dir = os.path.join(args.out, "onnx")
    pols = [os.path.abspath(args.baseline)]
    names = {os.path.basename(args.baseline): "baseline"}
    for c in ckpts:
        run = os.path.basename(os.path.dirname(os.path.abspath(c)))
        o = os.path.join(onnx_dir, f"{run}__{os.path.basename(c).replace('.pt', '.onnx')}")
        if not os.path.isfile(o):
            export(c, o)
        pols.append(o)
        names[os.path.basename(o)] = os.path.basename(o)
    cmd = [sys.executable, os.path.join(HERE, "mujoco_kick_stress.py"), "--suite", args.suite, "--trials", str(args.trials),
           "--workers", str(args.workers), "--out", os.path.join(args.out, "stress")]
    for p in pols:
        cmd += ["--policy", p]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    with open(os.path.join(args.out, "stress", "summary.json"), encoding="utf-8") as f:
        summ = json.load(f)["summary"]
    by = {}
    for s in summ:
        by.setdefault(s["policy"], {})[s["cond"]] = s
    base = by.get(os.path.basename(args.baseline), {})
    v_key = "train_fresh_obs" if args.suite == "gate" else "push_4N"
    v_ref = base.get(v_key, {}).get("peak_v_mean", 0.0)
    rows = []
    for pol, cells in by.items():
        falls = {k: v["fall_pct"] for k, v in cells.items()}
        kick = min(v["kick_pct"] for v in cells.values())
        v_nom = cells.get(v_key, {}).get("peak_v_mean", 0.0)
        flag = []
        if kick < 95.0:
            flag.append("kick<95%")
        if v_ref and v_nom < 0.85 * v_ref:
            flag.append("ball_speed<85%")
        rows.append(dict(policy=names.get(pol, pol), worst_fall=max(falls.values()),
                         mean_fall=sum(falls.values()) / len(falls), real_like=falls.get("real_like", -1),
                         real_like_low_mu=falls.get("real_like_low_mu", -1), min_kick=kick, v_nominal=v_nom,
                         flags=",".join(flag), **{f"fall_{k}": v for k, v in falls.items()}))
    rows.sort(key=lambda r: (bool(r["flags"]), r["worst_fall"], r["mean_fall"]))
    import csv
    with open(os.path.join(args.out, "ranking.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ["# MuJoCo gate ranking", "", f"trials per cell: {args.trials}. Lower is better. Simulation only.", ""]
    for r in rows:
        lines.append(f"- {r['policy']}: worst {r['worst_fall']:.1f} %, mean {r['mean_fall']:.1f} %, "
                     f"real_like {r['real_like']:.1f} %, low_mu {r['real_like_low_mu']:.1f} %, "
                     f"kick ≥ {r['min_kick']:.0f} %, v {r['v_nominal']:.2f} m/s {('⚠ ' + r['flags']) if r['flags'] else ''}")
    with open(os.path.join(args.out, "ranking.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
