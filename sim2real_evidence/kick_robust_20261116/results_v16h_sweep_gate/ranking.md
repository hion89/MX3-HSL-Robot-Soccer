# MuJoCo gate ranking

trials per cell: 300. Lower is better. Simulation only.

- baseline: worst 89.0 %, mean 31.5 %, real_like 78.3 %, low_mu 89.0 %, kick ≥ 100 %, v 4.55 m/s 
- ckpt_v16h__model_23800.onnx: worst 34.3 %, mean 20.9 %, real_like 14.7 %, low_mu 34.3 %, kick ≥ 95 %, v 3.86 m/s ⚠ ball_speed<85%
- ckpt_v16h__model_23899.onnx: worst 38.0 %, mean 17.6 %, real_like 14.7 %, low_mu 38.0 %, kick ≥ 94 %, v 3.71 m/s ⚠ kick<95%,ball_speed<85%
