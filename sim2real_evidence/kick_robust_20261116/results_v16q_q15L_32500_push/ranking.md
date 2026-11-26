# MuJoCo gate ranking

trials per cell: 150. Lower is better. Simulation only.

- baseline: worst 90.0 %, mean 38.1 %, real_like -1.0 %, low_mu -1.0 %, kick ≥ 100 %, v 4.34 m/s 
- ckpt_v16q__q15L_model_32500.onnx: worst 96.0 %, mean 91.9 %, real_like -1.0 %, low_mu -1.0 %, kick ≥ 18 %, v 1.53 m/s ⚠ kick<95%,ball_speed<85%
