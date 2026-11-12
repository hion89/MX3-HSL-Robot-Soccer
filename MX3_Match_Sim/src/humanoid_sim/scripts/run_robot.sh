#!/usr/bin/env bash
# 로봇 1 대 경기 스택 (자기 ROS_DOMAIN_ID 에서).
#   <워크스페이스>/src/humanoid_sim/scripts/run_robot.sh robo02 2 [launch 인자...]
# 정책 러너가 쓰는 onnxruntime 은 워크스페이스 파이썬 라이브러리에 있다 (sim_env.sh).
ROBOT=${1:?robot 이름 (예: robo02)}; DOMAIN=${2:?ROS_DOMAIN_ID}; shift 2
WS="${SIM_WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"   # src/humanoid_sim/scripts 에서 위로 3 단
source "$WS/src/humanoid_sim/scripts/sim_env.sh" || exit 1
export PYTHONPATH="$WS/src/humanoid_sim/pylib_sim:$PYTHONPATH"
# 정책 러너 onnxruntime 1 스레드·spinning 끔 (humanoid_sim/pylib_sim/sitecustomize.py). 비우면 원래대로: SIM_ORT_THREADS=
export SIM_ORT_THREADS=${SIM_ORT_THREADS-1}
export ROS_DOMAIN_ID=$DOMAIN
exec ros2 launch humanoid_sim sim_robot.launch.py robot:=$ROBOT "$@"
