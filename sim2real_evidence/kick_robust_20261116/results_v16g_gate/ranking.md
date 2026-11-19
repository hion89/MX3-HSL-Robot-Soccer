# MuJoCo gate ranking

trials per cell: 300. Lower is better. Simulation only.

- baseline: worst 89.0 %, mean 31.5 %, real_like 78.3 %, low_mu 89.0 %, kick ≥ 100 %, v 4.55 m/s 
- ckpt_v16g__model_24700.onnx: worst 22.3 %, mean 15.1 %, real_like 17.7 %, low_mu 22.3 %, kick ≥ 43 %, v 3.64 m/s ⚠ kick<95%,ball_speed<85%
- ckpt_v16g__model_24000.onnx: worst 26.7 %, mean 18.2 %, real_like 21.3 %, low_mu 26.7 %, kick ≥ 39 %, v 3.54 m/s ⚠ kick<95%,ball_speed<85%
