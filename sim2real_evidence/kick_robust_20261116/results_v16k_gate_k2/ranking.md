# MuJoCo gate ranking

trials per cell: 300. Lower is better. Simulation only.

- baseline: worst 89.0 %, mean 31.5 %, real_like 78.3 %, low_mu 89.0 %, kick ≥ 100 %, v 4.55 m/s 
- ckpt_v16k__k2_model_31300.onnx: worst 22.0 %, mean 12.5 %, real_like 5.7 %, low_mu 7.3 %, kick ≥ 87 %, v 3.44 m/s ⚠ kick<95%,ball_speed<85%
- ckpt_v16k__k2_model_31599.onnx: worst 53.3 %, mean 35.3 %, real_like 15.3 %, low_mu 19.0 %, kick ≥ 47 %, v 1.82 m/s ⚠ kick<95%,ball_speed<85%
