# MuJoCo gate ranking

trials per cell: 150. Lower is better. Simulation only.

- baseline: worst 90.0 %, mean 38.1 %, real_like -1.0 %, low_mu -1.0 %, kick ≥ 100 %, v 4.34 m/s 
- ckpt_v16q__q17_model_final.onnx: worst 48.0 %, mean 37.1 %, real_like -1.0 %, low_mu -1.0 %, kick ≥ 87 %, v 3.65 m/s ⚠ kick<95%,ball_speed<85%
