#!/usr/bin/env bash
# V1-6QC 한 단계: qc_train.sh MU ITERS LOAD_RUN CKPT RUN_NAME. 상태는 ~/kick16_runs/qc/RUN_NAME.status
MU=$1; IT=$2; LR=$3; CK=$4; RN=$5
source ~/anaconda3/etc/profile.d/conda.sh; conda activate env_kick16
mkdir -p ~/kick16_runs/qc; ST=~/kick16_runs/qc/$RN.status; : > $ST
cd ~/isaac_ws && git pull --ff-only -q; git log --oneline -1 >> $ST
cd ~/isaac_ws/unitree_rl_lab_kick16
export PYTHONPATH=$HOME/isaac_ws/unitree_rl_lab_kick16/source/unitree_rl_lab:$PYTHONPATH
export MX3_FEET_MU=$MU
E=unitree_mx3_walkkick_v1_6qc
mkdir -p logs/rsl_rl/$E/init_l3_31800
cp -n logs/rsl_rl/unitree_mx3_walkkick_v1_6l3/2026-11-26_13-18-36_sweep_from_k5/model_31800.pt logs/rsl_rl/$E/init_l3_31800/
while pgrep -f "scripts/rsl_rl/[t]rain.py" >/dev/null; do sleep 30; done
for try in 1 2 3; do
  L=~/kick16_runs/qc/${RN}_try$try.log
  ~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task Unitree-MX3-WalkKick-V1-6QC --experiment_name $E \
    --num_envs 8192 --max_iterations $IT --seed 42 --device cuda:0 --logger tensorboard --headless \
    --run_name $RN --resume --load_run $LR --checkpoint $CK > $L 2>&1 &
  pid=$! ok=0
  for i in $(seq 60); do sleep 10; grep -q "Learning iteration" $L && { ok=1; break; }; kill -0 $pid 2>/dev/null || break; done
  if [ $ok = 1 ]; then
    R=$(ls -d logs/rsl_rl/$E/2026-*_$RN | tail -1)
    echo "started $(date +%T) run $(basename $R)" >> $ST
    grep -A2 "static_friction_range" $R/params/env.yaml | tail -3 | tr '\n' ' ' >> $ST; echo >> $ST
    wait $pid; echo "done $(date +%T) rc=$?" >> $ST; exit 0
  fi
  echo "try$try no first iteration $(date +%T)" >> $ST
  grep -v "omni.usd\|shaderDefUtils" $L | grep -A12 Traceback | head -15 >> $ST
  pkill -9 -f "python scripts/rsl_rl/[t]rain.py --task Unitree-MX3-WalkKick-V1-6QC "; sleep 15
done
echo "failed" >> $ST
