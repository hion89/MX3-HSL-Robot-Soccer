# MuJoCo gate ranking

trials per cell: 300. Lower is better. Simulation only.

- baseline: worst 89.0 %, mean 31.5 %, real_like 78.3 %, low_mu 89.0 %, kick ≥ 100 %, v 4.55 m/s 
- ckpt_v16j__j3_model_31299.onnx: worst 33.7 %, mean 25.9 %, real_like 18.7 %, low_mu 17.7 %, kick ≥ 40 %, v 1.82 m/s ⚠ kick<95%,ball_speed<85%
- ckpt_v16j__j2_model_31299.onnx: worst 38.0 %, mean 28.8 %, real_like 19.3 %, low_mu 19.7 %, kick ≥ 70 %, v 3.19 m/s ⚠ kick<95%,ball_speed<85%
- ckpt_v16j__j4_model_31299.onnx: worst 39.7 %, mean 24.9 %, real_like 18.7 %, low_mu 21.7 %, kick ≥ 12 %, v 0.95 m/s ⚠ kick<95%,ball_speed<85%
- ckpt_v16j__j1_model_31299.onnx: worst 95.0 %, mean 82.6 %, real_like 68.7 %, low_mu 66.0 %, kick ≥ 23 %, v 1.23 m/s ⚠ kick<95%,ball_speed<85%
