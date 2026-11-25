# MuJoCo gate ranking

trials per cell: 300. Lower is better. Simulation only.

- baseline: worst 89.0 %, mean 31.5 %, real_like 78.3 %, low_mu 89.0 %, kick ≥ 100 %, v 4.55 m/s 
- ckpt_v16k__k1_model_31300.onnx: worst 51.3 %, mean 42.1 %, real_like 36.0 %, low_mu 34.0 %, kick ≥ 39 %, v 2.10 m/s ⚠ kick<95%,ball_speed<85%
- ckpt_v16k__k1_model_31599.onnx: worst 90.3 %, mean 61.9 %, real_like 36.0 %, low_mu 24.3 %, kick ≥ 17 %, v 0.81 m/s ⚠ kick<95%,ball_speed<85%
