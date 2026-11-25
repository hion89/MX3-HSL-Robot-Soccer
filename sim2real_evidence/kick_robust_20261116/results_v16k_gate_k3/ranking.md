# MuJoCo gate ranking

trials per cell: 300. Lower is better. Simulation only.

- baseline: worst 89.0 %, mean 31.5 %, real_like 78.3 %, low_mu 89.0 %, kick ≥ 100 %, v 4.55 m/s 
- ckpt_v16k__k3_model_31300.onnx: worst 19.7 %, mean 8.9 %, real_like 1.0 %, low_mu 8.0 %, kick ≥ 87 %, v 3.46 m/s ⚠ kick<95%,ball_speed<85%
- ckpt_v16k__k3_model_31599.onnx: worst 35.7 %, mean 20.8 %, real_like 8.3 %, low_mu 26.0 %, kick ≥ 95 %, v 3.54 m/s ⚠ kick<95%,ball_speed<85%
