#!/usr/bin/env bash
# 로봇에서 실행: 킥 정책 4개에 이름을 붙여 설치하고, 선택을 kick_push_31800 으로 바꾼다.
#   kick_power_22300    = V1-5B 22300 (원래 정책, 가장 셈, 실물에서 디딤발 미끄러짐·넘어짐)
#   kick_stable_27500   = V1-6H 27500 (넘어짐 가장 적음, 방향·속도 약함)
#   kick_accurate_30899 = V1-6I2 30899 (방향 22300 수준, 속도 83 %, 넘어짐 27500 과 비슷)
#   kick_push_31800     = V1-6L3 31800 (디딤발 쪽 외력에 가장 강함, 차는 다리 벌림 44°)
# 기존 파일 (walkkick_v1_5b_22300.onnx, policy_v16h_27500.onnx) 은 지우지 않는다.
# 다른 정책 선택: bash install_named_kicks.sh kick_stable_27500   (또는 kick_power_22300, kick_accurate_30899)
source /opt/ros/humble/setup.bash
source /home/robo04/sim2real_ws/install/setup.bash
set -eu
SEL=${1:-kick_push_31800}
IN=/home/robo04/incoming/robot_kick_models_20261126
WS=/home/robo04/sim2real_ws
LAUNCH=$WS/src/hsl_src/hsl_bringup/launch/real_role_trial.launch.py
MODELS=$WS/src/hsl_src/hsl_bringup/models
if pgrep -af '[a]ctor_onnx_walkkick_v1'; then echo "킥 러너가 실행 중입니다. start.sh 를 종료하고 다시 실행하세요."; exit 1; fi
test -f "$MODELS/$SEL.onnx" || test -f "$IN/$SEL.onnx" || { echo "모르는 정책 이름: $SEL"; exit 1; }
cd "$IN"; sha256sum -c SHA256SUMS
# 22300 은 로봇에 이미 있는 원본과 같은지 확인한다 (다르면 중단)
cmp -s "$IN/kick_power_22300.onnx" "$MODELS/walkkick_v1_5b_22300.onnx" || { echo "PC 의 22300 과 로봇의 walkkick_v1_5b_22300.onnx 가 다릅니다. 중단."; exit 1; }
cmp -s "$IN/kick_stable_27500.onnx" "$MODELS/policy_v16h_27500.onnx" || { echo "27500 파일이 로봇 설치본과 다릅니다. 중단."; exit 1; }
for f in kick_power_22300 kick_stable_27500 kick_accurate_30899; do cmp -s "$IN/$f.onnx" "$MODELS/$f.onnx" || { echo "$f 가 로봇 설치본과 다릅니다. 중단."; exit 1; }; done
BK=$(mktemp -d /home/robo04/incoming/kick_named_backup_XXXXXX); echo "BACKUP=$BK"
cp -- "$LAUNCH" "$BK/real_role_trial.launch.py"
for f in kick_power_22300 kick_stable_27500 kick_accurate_30899 kick_push_31800; do
  [ -e "$MODELS/$f.onnx" ] && cp -- "$MODELS/$f.onnx" "$BK/"
  cp -- "$IN/$f.onnx" "$MODELS/$f.onnx"
done
python3 - "$LAUNCH" "$SEL.onnx" <<'PY'
import re, sys
from pathlib import Path
p, new = Path(sys.argv[1]), sys.argv[2]
t = p.read_text(encoding='utf-8')
pat = re.compile(r"(kicking = os\.path\.join\(hsl, 'models', ')([^']+\.onnx)('\))")
if len(pat.findall(t)) != 1:
    raise SystemExit('kicking 선택 줄이 정확히 1개가 아닙니다. 변경하지 않았습니다.')
old = pat.search(t).group(2)
p.write_text(pat.sub(lambda m: m.group(1) + new + m.group(3), t), encoding='utf-8')
print('KICK MODEL:', old, '->', new)
PY
cd "$WS"; set +u
colcon build --base-paths src --symlink-install --packages-select hsl_bringup
source install/setup.bash; set -u
SH=$(ros2 pkg prefix --share hsl_bringup)
grep -n 'kicking =' "$SH/launch/real_role_trial.launch.py"
sha256sum "$SH/models/kick_"*.onnx
