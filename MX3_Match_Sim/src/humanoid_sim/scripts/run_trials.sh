#!/usr/bin/env bash
# 킥오프 정렬 튜닝 비교 (화면 없음). 예: run_trials.sh --configs base,lowmin,lowmin_tol --trials 8
WS="${SIM_WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"   # src/humanoid_sim/scripts 에서 위로 3 단
source "$WS/src/humanoid_sim/scripts/sim_env.sh" || exit 1
export ROS_DOMAIN_ID=${DOMAIN:-2}
exec python3 -m humanoid_sim.kick_trials "$@"
