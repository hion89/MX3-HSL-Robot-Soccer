# MuJoCo gate ranking

trials per cell: 300. Lower is better. Simulation only.

- baseline: worst 89.0 %, mean 31.5 %, real_like 78.3 %, low_mu 89.0 %, kick ≥ 100 %, v 4.55 m/s 
- ckpt_v16q__q17_model_final.onnx: worst 46.7 %, mean 26.1 %, real_like 37.7 %, low_mu 46.7 %, kick ≥ 80 %, v 3.62 m/s ⚠ kick<95%,ball_speed<85%
