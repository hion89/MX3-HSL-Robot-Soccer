#!/usr/bin/env bash
# Run one V1-6 experiment from model_22300 (inside the prepared unitree_rl_lab copy, Isaac Lab env active).
#
#   bash scripts/kick_v1_6/train_v16.sh C            # C = full (recommended first), or A / B for ablation
#   NUM_ENVS=8192 ITERS=2000 bash scripts/kick_v1_6/train_v16.sh A
#   SMOKE=1 bash scripts/kick_v1_6/train_v16.sh C    # 64 envs, 5 iterations: checks that the task builds
#
# Watch in TensorBoard: Curriculum/dc_blend (0.25 -> 1.0), Episode_Termination/fall_*,
#   Episode_Reward/stance_foot_yaw, metric_imu_delay_ms, metric_motor_volt, metric_turf_torque, ball_velocity.
# Checkpoints every save_interval; select with sim2real_evidence/kick_robust_20261116/gate_checkpoints.py,
# never by "last iteration".
set -euo pipefail
V=${1:?usage: train_v16.sh A|B|C}
case "$V" in A|B|C) ;; *) echo "variant must be A, B or C"; exit 2;; esac
LV=$(echo "$V" | tr 'A-Z' 'a-z')
NUM_ENVS=${NUM_ENVS:-16384}
ITERS=${ITERS:-3000}
[ -f "logs/rsl_rl/unitree_mx3_walkkick_v1_6$LV/init_22300/model_22300.pt" ] || {
  echo "missing resume checkpoint — run prepare_server_v16.sh first"; exit 3; }
if [ "${SMOKE:-0}" = "1" ]; then NUM_ENVS=64; ITERS=5; fi
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs/kick_v16
set -x
python scripts/rsl_rl/train.py --task "Unitree-MX3-WalkKick-V1-6$V" --headless --num_envs "$NUM_ENVS" \
  --resume --load_run init_22300 --checkpoint model_22300.pt --max_iterations "$ITERS" \
  2>&1 | tee "logs/kick_v16/train_${V}_$TS.log"
