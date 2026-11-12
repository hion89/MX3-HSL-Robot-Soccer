#!/usr/bin/env bash
# 경기 시뮬 시작 (설정은 README). Ctrl+C 로 전부 종료.
#   ./run_match.sh
#   SCENE=4v0 VIEWER=0 DURATION=180 ./run_match.sh
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="$WS/install/hsl_bringup/lib/hsl_bringup/run_match.sh"
[[ -x "$RUN" ]] || { echo "아직 빌드되지 않았습니다. 먼저 ./setup.sh 를 실행하세요" >&2; exit 1; }
SIM_WS="$WS" exec "$RUN" "$@"
