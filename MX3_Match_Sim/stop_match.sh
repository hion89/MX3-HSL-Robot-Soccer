#!/usr/bin/env bash
# 경기 시뮬 프로세스 정리 — 터미널을 그냥 닫았거나 Ctrl+C 뒤에도 뭔가 남았을 때.
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 로봇 스택은 ros2 launch 에 SIGINT 를 줘야 그 아래 노드까지 정리된다
pkill -INT -f "ros2 launch hsl_bringup sim_robot.launch.py"
pkill -f "humanoid_sim.sim_bridge"
pkill -f "$WS/install/hsl_bringup/lib/hsl_bringup/match_"
for _ in $(seq 1 20); do
  pgrep -f "$WS/install/|ros2 launch hsl_bringup sim_robot.launch.py|humanoid_sim.sim_bridge" > /dev/null || break
  sleep 0.5
done
# 그래도 남은 것 (이 폴더의 install 경로로 뜬 노드) 강제 종료
pkill -9 -f "$WS/install/" && echo "남은 노드를 강제 종료했습니다"
pgrep -fa "humanoid_sim.sim_bridge|ros2 launch hsl_bringup" || echo "정리 완료"
