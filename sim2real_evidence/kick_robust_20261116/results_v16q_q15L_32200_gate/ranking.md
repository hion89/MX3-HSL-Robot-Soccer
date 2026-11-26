# MuJoCo gate ranking

trials per cell: 300. Lower is better. Simulation only.

- baseline: worst 89.0 %, mean 31.5 %, real_like 78.3 %, low_mu 89.0 %, kick ≥ 100 %, v 4.55 m/s 
- ckpt_v16q__q15L_model_32200.onnx: worst 53.0 %, mean 30.0 %, real_like 32.7 %, low_mu 53.0 %, kick ≥ 89 %, v 3.69 m/s ⚠ kick<95%,ball_speed<85%
