# MuJoCo gate ranking

trials per cell: 300. Lower is better. Simulation only.

- baseline: worst 89.0 %, mean 31.5 %, real_like 78.3 %, low_mu 89.0 %, kick ≥ 100 %, v 4.55 m/s 
- ckpt_v16q__q17_model_32200.onnx: worst 43.0 %, mean 22.8 %, real_like 23.7 %, low_mu 43.0 %, kick ≥ 92 %, v 3.66 m/s ⚠ kick<95%,ball_speed<85%
