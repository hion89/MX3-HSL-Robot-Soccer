# MuJoCo gate ranking

trials per cell: 300. Lower is better. Simulation only.

- baseline: worst 89.0 %, mean 31.5 %, real_like 78.3 %, low_mu 89.0 %, kick ≥ 100 %, v 4.55 m/s 
- ckpt_v16q__q13f_model_32200.onnx: worst 73.7 %, mean 55.6 %, real_like 44.7 %, low_mu 39.7 %, kick ≥ 13 %, v 1.01 m/s ⚠ kick<95%,ball_speed<85%
