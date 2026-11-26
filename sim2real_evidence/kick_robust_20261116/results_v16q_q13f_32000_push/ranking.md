# MuJoCo gate ranking

trials per cell: 150. Lower is better. Simulation only.

- baseline: worst 90.0 %, mean 38.1 %, real_like -1.0 %, low_mu -1.0 %, kick ≥ 100 %, v 4.34 m/s 
- ckpt_v16q__q13f_model_32000.onnx: worst 50.0 %, mean 28.7 %, real_like -1.0 %, low_mu -1.0 %, kick ≥ 60 %, v 2.80 m/s ⚠ kick<95%,ball_speed<85%
