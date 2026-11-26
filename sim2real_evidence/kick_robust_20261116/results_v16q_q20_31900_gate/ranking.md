# MuJoCo gate ranking

trials per cell: 300. Lower is better. Simulation only.

- baseline: worst 89.0 %, mean 31.5 %, real_like 78.3 %, low_mu 89.0 %, kick ≥ 100 %, v 4.55 m/s 
- ckpt_v16q__q20_model_31900.onnx: worst 15.3 %, mean 8.6 %, real_like 1.7 %, low_mu 7.7 %, kick ≥ 79 %, v 3.41 m/s ⚠ kick<95%,ball_speed<85%
