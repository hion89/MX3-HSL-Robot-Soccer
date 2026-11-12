#!/usr/bin/env bash
# 시뮬 경기 — MuJoCo 브리지 1 개 + 로봇마다 스택 1 개 (도메인 분리) + 간이 GC + 모니터. Ctrl-C 로 전부 종료.
#   run_match.sh                 # 4v4, 뷰어, 5 분
#   SCENE=4v0 DURATION=180 VIEWER=0 run_match.sh
# 장면: hsl_bringup/models/scene_<SCENE>.{xml,json}  (로봇 이름 t<팀>p<번호>, 도메인 = 장면 json)
# 심판(match_referee.py): initial 3 s → ready READY_S → set 5 s → playing. 골 → 센터 재배치·실점 팀 킥오프, 아웃 → 안쪽 재배치
WS="${SIM_WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"   # install/hsl_bringup/lib/hsl_bringup 에서 위로 4 단
source "$WS/src/humanoid_sim/scripts/sim_env.sh" || exit 1
# 이미 돌고 있는 경기가 있으면 시작하지 않는다 (같은 도메인에 브리지·심판이 둘이면 서로 얽힌다)
if ps -eo args | awk '$1 ~ /python3$/ && $2=="-m" && $3=="humanoid_sim.sim_bridge"' | grep -q .; then
  echo "[match] 이미 시뮬 브리지가 돌고 있습니다. 먼저 끄세요 (그래도 띄우려면 FORCE=1)." >&2
  [[ "${FORCE:-0}" == 1 ]] || exit 1
fi
# 강제 종료로 남은 Fast DDS 공유 메모리 찌꺼기 정리 (쌓이면 일부 로봇의 메시지 전달이 끊긴다 — 파랑 4 번 정지 사례)
fastdds shm clean > /dev/null 2>&1 || true
SCENE=${SCENE:-4v4}; VIEWER=${VIEWER:-1}; DURATION=${DURATION:-300}; READY_S=${READY_S:-40}
KICKOFF=${KICKOFF:-35}
STACK_ARGS=${STACK_ARGS:-}   # 로봇 스택 launch 인자 덧붙이기 (예: STACK_ARGS="team_send_rate_hz:=4.0")
SHARE="$(ros2 pkg prefix hsl_bringup)/share/hsl_bringup"
LIB="$(ros2 pkg prefix hsl_bringup)/lib/hsl_bringup"
SC="$SHARE/models/scene_$SCENE"
LOG="${LOG:-$WS/logs/match_${SCENE}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG"
trap 'trap - EXIT INT TERM; echo; echo "[match] 종료 중…"; kill 0' EXIT INT TERM

BR_ARGS=(--scene "$SC")
[[ "$VIEWER" == 1 ]] && BR_ARGS+=(--viewer)
"$WS/src/humanoid_sim/scripts/run_bridge.sh" "${BR_ARGS[@]}" "$@" > "$LOG/bridge.log" 2>&1 &
sleep 3
ROBOTS=$(python3 -c "
import json,re
for r in json.load(open('$SC.json'))['robots']:
    m=re.match(r't(\d+)p(\d+)',r['name']); print(r['name'],r['domain'],m.group(1),m.group(2))")
while read -r NAME DOM TEAM PL; do
  "$LIB/run_stack.sh" "$DOM" "$TEAM" "$PL" node_output:=log $STACK_ARGS > "$LOG/$NAME.log" 2>&1 &
  sleep 0.5
done <<< "$ROBOTS"
TEAMS=$(echo "$ROBOTS" | awk '{print $3}' | sort -u | tr '\n' ' ')
T1=$(echo $TEAMS | awk '{print $1}'); T2=$(echo $TEAMS | awk '{print $2}'); T2=${T2:-$((T1 + 1))}
echo "[match] 장면 $SCENE (팀 $TEAMS), 로그 $LOG — 스택 올라오는 중 (20 s)"
sleep 20
while read -r NAME DOM TEAM PL; do
  ROS_DOMAIN_ID=$DOM timeout 10 ros2 service call /sim/start std_srvs/srv/Trigger > /dev/null && echo "[match] $NAME START" &
done <<< "$ROBOTS"
sleep 6
# 건강 검사: START 뒤 보행 러너가 정책을 돌리고 있는지 (로그의 tick=). 안 돌면 경고만 (재시작은 사람 판단)
sleep 4
while read -r NAME DOM TEAM PL; do
  grep -q "actor_onnx_walking_node\]: tick=" "$LOG/$NAME.log" 2>/dev/null || \
    echo "[match] ⚠ $NAME 보행 러너가 아직 정책을 안 돌림 (통신 고립 의심) — $LOG/$NAME.log"
done <<< "$ROBOTS"
DOM0=$(echo "$ROBOTS" | head -1 | awk '{print $2}')
python3 "$LIB/match_referee.py" --team "$T1" --opp "$T2" --kickoff "$KICKOFF" --domain "$DOM0" \
  --ready "$READY_S" --half "${HALF:-600}" > "$LOG/referee.log" 2>&1 &
echo "[match] 심판/GC: initial 3 s → ready ${READY_S} s → set 5 s → playing (킥오프 팀 $KICKOFF), 골·아웃 자동 처리"
python3 "$LIB/match_monitor.py" --scene "$SC" --duration "$DURATION" --period 10 --log "$LOG/monitor.jsonl" \
  | tee "$LOG/monitor.txt"
