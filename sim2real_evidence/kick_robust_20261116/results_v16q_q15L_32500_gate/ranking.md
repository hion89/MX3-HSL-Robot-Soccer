# MuJoCo gate ranking

trials per cell: 300. Lower is better. Simulation only.

- baseline: worst 89.0 %, mean 31.5 %, real_like 78.3 %, low_mu 89.0 %, kick ≥ 100 %, v 4.55 m/s 
- ckpt_v16q__q15L_model_32500.onnx: worst 99.0 %, mean 96.6 %, real_like 92.0 %, low_mu 91.0 %, kick ≥ 2 %, v 0.65 m/s ⚠ kick<95%,ball_speed<85%
