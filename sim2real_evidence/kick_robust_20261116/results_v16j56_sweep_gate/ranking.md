# MuJoCo gate ranking

trials per cell: 300. Lower is better. Simulation only.

- baseline: worst 89.0 %, mean 31.5 %, real_like 78.3 %, low_mu 89.0 %, kick ≥ 100 %, v 4.55 m/s 
- ckpt_v16j__j6_model_31599.onnx: worst 39.7 %, mean 26.2 %, real_like 34.0 %, low_mu 39.7 %, kick ≥ 25 %, v 1.54 m/s ⚠ kick<95%,ball_speed<85%
- ckpt_v16j__j5_model_31599.onnx: worst 97.0 %, mean 93.5 %, real_like 90.3 %, low_mu 84.7 %, kick ≥ 8 %, v 0.54 m/s ⚠ kick<95%,ball_speed<85%
