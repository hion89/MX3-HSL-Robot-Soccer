#!/usr/bin/env bash
# 눈으로 보는 데모 — MuJoCo 뷰어(실제) + RViz(스택의 추정 vs 정답) + 모니터. Ctrl-C 로 전부 종료.
#   <워크스페이스>/src/humanoid_sim/scripts/run_demo.sh
# RViz: 초록 원기둥·화살표 = 실제 로봇, 흰 공 = 실제 공 / TF base_link·VisionPose·EKF = 스택이 생각하는 위치
# 순서: 스택 기동 12 s → START → GC READY → 전략이 POSITION_DONE(포메이션 도착) 되면 곧바로 SET 1 s → PLAYING
# TUNED=1 (기본): 시뮬 비교에서 좋았던 설정을 적용 (원본 yaml 은 그대로, 이 실행에서만)
#   목 yaw 고정 해제 (neck_yaw_lock_deg:=-71), visual servo 최소속도 vx·후진·vy 0.08 / wz 0.15,
#   visual servo GETBEHIND 교착 수정 (SIM_SERVO_FIX=1)
#   TUNED=0 이면 원래 설정 그대로
WS="${SIM_WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"   # src/humanoid_sim/scripts 에서 위로 3 단
source "$WS/src/humanoid_sim/scripts/sim_env.sh" || exit 1
export ROS_DOMAIN_ID=${DOMAIN:-2}
H="$WS/src/humanoid_sim/scripts"
SHARE="$WS/install/humanoid_sim/share/humanoid_sim"
LOG="$WS/logs/demo_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG"
trap 'trap - EXIT INT TERM; echo; echo "[demo] 종료 중…"; kill 0' EXIT INT TERM

"$H/run_bridge.sh" --scene "$SHARE/models/scene_1v0" --viewer "$@" > "$LOG/bridge.log" 2>&1 &
sleep 2
TUNED=${TUNED:-1}
LAUNCH_ARGS=()
if [[ "$TUNED" == 1 ]]; then
  LAUNCH_ARGS+=("neck_yaw_lock_deg:=-71")
  export SIM_SERVO_FIX=1   # visual servo GETBEHIND 교착 수정 (pylib_sim/sitecustomize.py, 원본 소스는 그대로)
fi
"$H/run_robot.sh" robo02 "$ROS_DOMAIN_ID" "${LAUNCH_ARGS[@]}" > "$LOG/stack.log" 2>&1 &
rviz2 -d "$SHARE/config/sim_view.rviz" > "$LOG/rviz.log" 2>&1 &

# GameController 는 FIFO 로 상태 명령을 받는다 (initial / ready / set / playing)
mkfifo "$LOG/gc_cmd"
ros2 run humanoid_sim sim_gc --state initial < "$LOG/gc_cmd" > "$LOG/gc.log" 2>&1 &
exec 3> "$LOG/gc_cmd"

echo "[demo] 스택 올라오는 중 (12 s)… 로그: $LOG"
sleep 12
ros2 service call /sim/start std_srvs/srv/Trigger > /dev/null && echo "[demo] START"
if [[ "$TUNED" == 1 ]]; then
  for kv in "vx_min 0.08" "vx_back_min 0.08" "vy_min 0.08" "wz_min 0.15"; do
    ros2 param set /visual_servo_kick_node $kv > /dev/null
  done
  echo "[demo] 개선 설정 적용: 목 yaw 고정 해제, servo 최소속도 0.08 / wz 0.15, GETBEHIND 교착 수정"
else
  echo "[demo] 원래 설정 그대로 (TUNED=0)"
fi
ros2 run humanoid_sim sim_monitor &
sleep 2
echo ready >&3
echo "[demo] GC READY — 포메이션 위치로 이동"
if timeout 120 ros2 topic echo /strategy/state --field strategy_phase 2>/dev/null | grep -m1 -q POSITION_DONE; then
  echo set >&3
  echo "[demo] 포메이션 도착 → GC SET"
  sleep 1
  echo playing >&3
  echo "[demo] GC PLAYING — 킥오프!"
else
  echo "[demo] 120 s 안에 POSITION_DONE 이 안 와서 그대로 PLAYING"
  echo playing >&3
fi
wait
