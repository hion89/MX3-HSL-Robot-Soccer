#!/usr/bin/env bash
# 시뮬 로봇 1 대 스택 (자기 ROS_DOMAIN_ID 에서): mx3_src 하위 계층 + hsl 상위 계층.
#   run_stack.sh <domain> <team> <player> [launch 인자...]
DOMAIN=${1:?domain}; TEAM=${2:?team}; PLAYER=${3:?player}; shift 3
WS="${SIM_WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"   # install/hsl_bringup/lib/hsl_bringup 에서 위로 4 단
source "$WS/src/humanoid_sim/scripts/sim_env.sh" || exit 1
# 시뮬 전용 패치 (정책 러너 onnxruntime 1 스레드)
export PYTHONPATH="$WS/src/humanoid_sim/pylib_sim:$PYTHONPATH"
export SIM_ORT_THREADS=${SIM_ORT_THREADS-1}
export ROS_DOMAIN_ID=$DOMAIN
export PYTHONUNBUFFERED=1
exec ros2 launch hsl_bringup sim_robot.launch.py team_number:=$TEAM player_number:=$PLAYER "$@"
