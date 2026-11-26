# MuJoCo gate ranking

trials per cell: 300. Lower is better. Simulation only.

- baseline: worst 89.0 %, mean 31.5 %, real_like 78.3 %, low_mu 89.0 %, kick ≥ 100 %, v 4.55 m/s 
- ckpt_v16q__q15L_model_32000.onnx: worst 55.0 %, mean 25.9 %, real_like 33.7 %, low_mu 55.0 %, kick ≥ 94 %, v 3.70 m/s ⚠ kick<95%,ball_speed<85%
