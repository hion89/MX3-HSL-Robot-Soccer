# MuJoCo gate ranking

trials per cell: 300. Lower is better. Simulation only.

- baseline: worst 89.0 %, mean 31.5 %, real_like 78.3 %, low_mu 89.0 %, kick ≥ 100 %, v 4.55 m/s 
- ckpt_v16q__q20_model_32000.onnx: worst 21.3 %, mean 13.8 %, real_like 15.7 %, low_mu 21.3 %, kick ≥ 77 %, v 3.56 m/s ⚠ kick<95%,ball_speed<85%
