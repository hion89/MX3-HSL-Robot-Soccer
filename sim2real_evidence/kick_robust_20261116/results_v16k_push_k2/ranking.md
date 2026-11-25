# MuJoCo gate ranking

trials per cell: 150. Lower is better. Simulation only.

- baseline: worst 90.0 %, mean 38.1 %, real_like -1.0 %, low_mu -1.0 %, kick ≥ 100 %, v 4.34 m/s 
- ckpt_v16k__k2_model_31300.onnx: worst 91.3 %, mean 51.9 %, real_like -1.0 %, low_mu -1.0 %, kick ≥ 88 %, v 3.73 m/s ⚠ kick<95%
- ckpt_v16k__k2_model_31599.onnx: worst 93.3 %, mean 52.4 %, real_like -1.0 %, low_mu -1.0 %, kick ≥ 57 %, v 2.15 m/s ⚠ kick<95%,ball_speed<85%
