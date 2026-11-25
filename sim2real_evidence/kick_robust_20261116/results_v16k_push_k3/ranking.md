# MuJoCo gate ranking

trials per cell: 150. Lower is better. Simulation only.

- baseline: worst 90.0 %, mean 38.1 %, real_like -1.0 %, low_mu -1.0 %, kick ≥ 100 %, v 4.34 m/s 
- ckpt_v16k__k3_model_31599.onnx: worst 50.7 %, mean 16.8 %, real_like -1.0 %, low_mu -1.0 %, kick ≥ 98 %, v 3.63 m/s ⚠ ball_speed<85%
- ckpt_v16k__k3_model_31300.onnx: worst 83.3 %, mean 36.0 %, real_like -1.0 %, low_mu -1.0 %, kick ≥ 91 %, v 3.68 m/s ⚠ kick<95%,ball_speed<85%
