# MuJoCo gate ranking

trials per cell: 150. Lower is better. Simulation only.

- baseline: worst 88.0 %, mean 31.0 %, real_like 76.0 %, low_mu 88.0 %, kick ≥ 100 %, v 4.54 m/s 
- ckpt_v16j__j5_model_31200.onnx: worst 20.0 %, mean 12.3 %, real_like 4.7 %, low_mu 7.3 %, kick ≥ 88 %, v 3.81 m/s ⚠ kick<95%,ball_speed<85%
- ckpt_v16j__j5_model_31100.onnx: worst 20.0 %, mean 13.0 %, real_like 2.7 %, low_mu 11.3 %, kick ≥ 89 %, v 3.85 m/s ⚠ kick<95%,ball_speed<85%
- ckpt_v16j__j5_model_31300.onnx: worst 23.3 %, mean 12.8 %, real_like 2.7 %, low_mu 8.7 %, kick ≥ 83 %, v 3.71 m/s ⚠ kick<95%,ball_speed<85%
- ckpt_v16j__j6_model_31200.onnx: worst 23.3 %, mean 17.2 %, real_like 10.0 %, low_mu 15.3 %, kick ≥ 83 %, v 3.48 m/s ⚠ kick<95%,ball_speed<85%
- ckpt_v16j__j6_model_31100.onnx: worst 29.3 %, mean 18.4 %, real_like 5.3 %, low_mu 15.3 %, kick ≥ 92 %, v 3.64 m/s ⚠ kick<95%,ball_speed<85%
- ckpt_v16j__j6_model_31300.onnx: worst 32.7 %, mean 24.2 %, real_like 16.0 %, low_mu 32.7 %, kick ≥ 89 %, v 3.52 m/s ⚠ kick<95%,ball_speed<85%
