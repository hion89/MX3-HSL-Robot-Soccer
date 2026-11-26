# MuJoCo gate ranking

trials per cell: 300. Lower is better. Simulation only.

- baseline: worst 89.0 %, mean 31.5 %, real_like 78.3 %, low_mu 89.0 %, kick ≥ 100 %, v 4.55 m/s 
- ckpt_v16q__q15L_model_33500.onnx: worst 94.3 %, mean 61.6 %, real_like 63.7 %, low_mu 69.7 %, kick ≥ 0 %, v 0.00 m/s ⚠ kick<95%,ball_speed<85%
