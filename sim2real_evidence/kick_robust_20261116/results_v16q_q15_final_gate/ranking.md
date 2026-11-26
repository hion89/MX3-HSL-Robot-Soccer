# MuJoCo gate ranking

trials per cell: 300. Lower is better. Simulation only.

- baseline: worst 89.0 %, mean 31.5 %, real_like 78.3 %, low_mu 89.0 %, kick ≥ 100 %, v 4.55 m/s 
- ckpt_v16q__q15_model_final.onnx: worst 35.7 %, mean 25.7 %, real_like 16.3 %, low_mu 33.0 %, kick ≥ 91 %, v 3.55 m/s ⚠ kick<95%,ball_speed<85%
