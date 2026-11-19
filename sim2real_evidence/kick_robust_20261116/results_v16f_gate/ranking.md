# MuJoCo gate ranking

trials per cell: 300. Lower is better. Simulation only.

- baseline: worst 89.0 %, mean 31.5 %, real_like 78.3 %, low_mu 89.0 %, kick ≥ 100 %, v 4.55 m/s 
- ckpt_v16f__model_23500.onnx: worst 11.0 %, mean 3.0 %, real_like 1.0 %, low_mu 4.0 %, kick ≥ 62 %, v 3.89 m/s ⚠ kick<95%
- ckpt_v16f__model_23000.onnx: worst 20.3 %, mean 12.7 %, real_like 11.3 %, low_mu 17.7 %, kick ≥ 58 %, v 3.95 m/s ⚠ kick<95%
- ckpt_v16f__model_23400.onnx: worst 46.0 %, mean 28.9 %, real_like 40.3 %, low_mu 45.7 %, kick ≥ 64 %, v 3.90 m/s ⚠ kick<95%
