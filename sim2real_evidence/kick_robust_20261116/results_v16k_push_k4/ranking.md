# MuJoCo gate ranking

trials per cell: 150. Lower is better. Simulation only.

- baseline: worst 90.0 %, mean 38.1 %, real_like -1.0 %, low_mu -1.0 %, kick ≥ 100 %, v 4.34 m/s 
- ckpt_v16k__k4_model_31300.onnx: worst 79.3 %, mean 34.3 %, real_like -1.0 %, low_mu -1.0 %, kick ≥ 62 %, v 2.32 m/s ⚠ kick<95%,ball_speed<85%
- ckpt_v16k__k4_model_31599.onnx: worst 100.0 %, mean 100.0 %, real_like -1.0 %, low_mu -1.0 %, kick ≥ 3 %, v 0.24 m/s ⚠ kick<95%,ball_speed<85%
