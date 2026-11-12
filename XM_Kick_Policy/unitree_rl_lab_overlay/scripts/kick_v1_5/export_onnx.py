"""rsl_rl 체크포인트 (actor MLP) → ONNX. Isaac 없이 torch 만으로 — 학습 중 MuJoCo 표준 시험용.

play.py 의 내보내기와 같은 형태: 입력 "obs" [1, 265], 출력 "actions" [1, 12], ELU, 정규화 없음
(WalkKickPPORunnerCfg: empirical_normalization false, actor_hidden_dims 512/256/128).

  conda activate isaac_env
  python scripts/kick_v1_5/export_onnx.py logs/rsl_rl/unitree_mx3_walkkick_v1_5/<run>/model_18700.pt
  (--check ref.onnx 는 onnxruntime 이 있는 환경에서. 18200 → 실기 walkkick_v1_policy.onnx 와 최대 차 0.0 확인)
결과: 체크포인트 옆 exported_mj/model_18700.onnx
"""
import argparse
import os

import torch
from torch import nn

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--out", default=None)
ap.add_argument("--check", default=None, help="같은 가중치 ONNX 와 출력 비교 (onnxruntime)")
a = ap.parse_args()

sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
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
n_in = layers[0].in_features
out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.ckpt)), "exported_mj",
                            os.path.basename(a.ckpt).replace(".pt", ".onnx"))
os.makedirs(os.path.dirname(out), exist_ok=True)
torch.onnx.export(net, torch.zeros(1, n_in), out, input_names=["obs"], output_names=["actions"], opset_version=18,
                  dynamo=False)
print(f"{a.ckpt} -> {out}  (obs {n_in}, actions {layers[-1].out_features})")
if a.check:
    import numpy as np
    import onnxruntime as ort
    x = np.random.default_rng(0).normal(size=(64, n_in)).astype(np.float32)
    r = [ort.InferenceSession(p, providers=["CPUExecutionProvider"]) for p in (out, a.check)]
    ya = np.concatenate([r[0].run(None, {"obs": x[i:i + 1]})[0] for i in range(64)])
    yb = np.concatenate([r[1].run(None, {"obs": x[i:i + 1]})[0] for i in range(64)])
    print(f"기준 ONNX 와 최대 차 {np.abs(ya - yb).max():.2e}")
