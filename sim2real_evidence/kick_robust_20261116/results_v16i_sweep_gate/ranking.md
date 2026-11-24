# MuJoCo gate ranking

trials per cell: 300. Lower is better. Simulation only.

- baseline: worst 89.0 %, mean 31.5 %, real_like 78.3 %, low_mu 89.0 %, kick ≥ 100 %, v 4.55 m/s 
- ckpt_v16i__i_model_27900.onnx: worst 10.3 %, mean 5.3 %, real_like 2.3 %, low_mu 7.7 %, kick ≥ 74 %, v 2.97 m/s ⚠ kick<95%,ball_speed<85%
- ckpt_v16i__i2_model_27900.onnx: worst 45.7 %, mean 24.9 %, real_like 17.0 %, low_mu 45.7 %, kick ≥ 99 %, v 3.73 m/s ⚠ ball_speed<85%
