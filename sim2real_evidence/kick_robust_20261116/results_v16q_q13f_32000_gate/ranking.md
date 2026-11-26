# MuJoCo gate ranking

trials per cell: 300. Lower is better. Simulation only.

- baseline: worst 89.0 %, mean 31.5 %, real_like 78.3 %, low_mu 89.0 %, kick ≥ 100 %, v 4.55 m/s 
- ckpt_v16q__q13f_model_32000.onnx: worst 34.0 %, mean 21.1 %, real_like 23.0 %, low_mu 29.7 %, kick ≥ 58 %, v 2.47 m/s ⚠ kick<95%,ball_speed<85%
