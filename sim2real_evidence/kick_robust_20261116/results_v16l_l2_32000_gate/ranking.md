# MuJoCo gate ranking

trials per cell: 300. Lower is better. Simulation only.

- baseline: worst 89.0 %, mean 31.5 %, real_like 78.3 %, low_mu 89.0 %, kick ≥ 100 %, v 4.55 m/s 
- ckpt_v16l__l2_model_32000.onnx: worst 26.3 %, mean 18.2 %, real_like 6.3 %, low_mu 24.3 %, kick ≥ 91 %, v 3.63 m/s ⚠ kick<95%,ball_speed<85%
