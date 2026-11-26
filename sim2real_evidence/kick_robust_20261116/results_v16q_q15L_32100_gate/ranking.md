# MuJoCo gate ranking

trials per cell: 300. Lower is better. Simulation only.

- baseline: worst 89.0 %, mean 31.5 %, real_like 78.3 %, low_mu 89.0 %, kick ≥ 100 %, v 4.55 m/s 
- ckpt_v16q__q15L_model_32100.onnx: worst 42.7 %, mean 27.7 %, real_like 25.3 %, low_mu 42.7 %, kick ≥ 93 %, v 3.76 m/s ⚠ kick<95%,ball_speed<85%
