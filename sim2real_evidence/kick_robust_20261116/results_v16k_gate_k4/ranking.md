# MuJoCo gate ranking

trials per cell: 300. Lower is better. Simulation only.

- baseline: worst 89.0 %, mean 31.5 %, real_like 78.3 %, low_mu 89.0 %, kick ≥ 100 %, v 4.55 m/s 
- ckpt_v16k__k4_model_31300.onnx: worst 85.7 %, mean 63.3 %, real_like 61.0 %, low_mu 67.3 %, kick ≥ 39 %, v 1.58 m/s ⚠ kick<95%,ball_speed<85%
- ckpt_v16k__k4_model_31599.onnx: worst 100.0 %, mean 99.9 %, real_like 100.0 %, low_mu 100.0 %, kick ≥ 1 %, v 0.14 m/s ⚠ kick<95%,ball_speed<85%
