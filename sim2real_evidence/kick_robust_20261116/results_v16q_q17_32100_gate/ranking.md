# MuJoCo gate ranking

trials per cell: 300. Lower is better. Simulation only.

- baseline: worst 89.0 %, mean 31.5 %, real_like 78.3 %, low_mu 89.0 %, kick ≥ 100 %, v 4.55 m/s 
- ckpt_v16q__q17_model_32100.onnx: worst 29.0 %, mean 18.3 %, real_like 13.3 %, low_mu 29.0 %, kick ≥ 93 %, v 3.85 m/s ⚠ kick<95%,ball_speed<85%
