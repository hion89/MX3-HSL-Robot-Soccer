# MuJoCo gate ranking

trials per cell: 300. Lower is better. Simulation only.

- baseline: worst 89.0 %, mean 31.5 %, real_like 78.3 %, low_mu 89.0 %, kick ≥ 100 %, v 4.55 m/s 
- ckpt_v16q__q13f_model_31900.onnx: worst 25.0 %, mean 13.0 %, real_like 16.0 %, low_mu 25.0 %, kick ≥ 76 %, v 3.32 m/s ⚠ kick<95%,ball_speed<85%
