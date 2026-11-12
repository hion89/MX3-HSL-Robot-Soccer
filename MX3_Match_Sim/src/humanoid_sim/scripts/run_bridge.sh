#!/usr/bin/env bash
# MuJoCo 브리지 실행. 시스템 Python 3.12 + ROS Jazzy + 워크스페이스 파이썬 라이브러리(mujoco, sim_env.sh).
#   <워크스페이스>/src/humanoid_sim/scripts/run_bridge.sh --scene <장면 경로, 확장자 없이> [--viewer]
WS="${SIM_WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"   # src/humanoid_sim/scripts 에서 위로 3 단
source "$WS/src/humanoid_sim/scripts/sim_env.sh" || exit 1
exec python3 -m humanoid_sim.sim_bridge "$@"
