# MuJoCo gate ranking

trials per cell: 150. Lower is better. Simulation only.

- baseline: worst 90.0 %, mean 38.1 %, real_like -1.0 %, low_mu -1.0 %, kick ≥ 100 %, v 4.34 m/s 
- ckpt_v16q__q13f_model_32100.onnx: worst 42.0 %, mean 36.4 %, real_like -1.0 %, low_mu -1.0 %, kick ≥ 29 %, v 1.95 m/s ⚠ kick<95%,ball_speed<85%
