# MuJoCo gate ranking

trials per cell: 150. Lower is better. Simulation only.

- baseline: worst 90.0 %, mean 38.1 %, real_like -1.0 %, low_mu -1.0 %, kick ≥ 100 %, v 4.34 m/s 
- ckpt_v16k__k1_model_31300.onnx: worst 76.7 %, mean 62.7 %, real_like -1.0 %, low_mu -1.0 %, kick ≥ 49 %, v 2.53 m/s ⚠ kick<95%,ball_speed<85%
- ckpt_v16k__k1_model_31599.onnx: worst 91.3 %, mean 73.5 %, real_like -1.0 %, low_mu -1.0 %, kick ≥ 21 %, v 0.94 m/s ⚠ kick<95%,ball_speed<85%
