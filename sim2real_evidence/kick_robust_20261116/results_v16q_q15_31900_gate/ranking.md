# MuJoCo gate ranking

trials per cell: 300. Lower is better. Simulation only.

- baseline: worst 89.0 %, mean 31.5 %, real_like 78.3 %, low_mu 89.0 %, kick ≥ 100 %, v 4.55 m/s 
- ckpt_v16q__q15_model_31900.onnx: worst 18.0 %, mean 9.8 %, real_like 5.7 %, low_mu 18.0 %, kick ≥ 95 %, v 3.51 m/s ⚠ kick<95%,ball_speed<85%
