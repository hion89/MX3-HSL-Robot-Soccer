# MuJoCo gate ranking

trials per cell: 300. Lower is better. Simulation only.

- baseline: worst 89.0 %, mean 31.5 %, real_like 78.3 %, low_mu 89.0 %, kick ≥ 100 %, v 4.55 m/s 
- ckpt_v16k__k5_model_31599.onnx: worst 24.0 %, mean 13.3 %, real_like 9.0 %, low_mu 24.0 %, kick ≥ 89 %, v 3.78 m/s ⚠ kick<95%,ball_speed<85%
- ckpt_v16k__k5_model_31300.onnx: worst 56.3 %, mean 30.0 %, real_like 36.7 %, low_mu 56.3 %, kick ≥ 90 %, v 3.78 m/s ⚠ kick<95%,ball_speed<85%
